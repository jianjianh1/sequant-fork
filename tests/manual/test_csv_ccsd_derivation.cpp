// Derives the REAL closed-shell CSV/PNO CCSD residual equations via the
// FULL SeQuant pipeline MPQC's own cck.ipp/sequant.cpp uses, matching
// CCk<Tile,Policy>::generate_csv_closedshell() (cck.ipp:1507-1541) exactly:
//   Rs = make_cceqvec_csv_closedshell(k, zero_t1)
//   for r != 0: e = tail_factor(e)          // second strip -- biorg() re-added S
//   if (df_):   e = density_fit(e, dfbs, "g", "g")
//   e = csv_transform(e, pao_uocc, "C")
//   if (seq_opt_): flatten(e); e = optimize(e, opts)
//
// Phase 0/1 of twinkly-dazzling-shamir.md: confirm this reproduces the
// "only C/t carry proto_indices" invariant, then feed the result into
// TiledArrayGenerator.
#include <SeQuant/core/context.hpp>
#include <SeQuant/core/export/export.hpp>
#include <SeQuant/core/export/tiledarray_generator.hpp>
#include <SeQuant/core/expressions/expr_algorithms.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/core/io/serialization/serialization.hpp>
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/core/optimize/common_subexpression_elimination.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/core/rational.hpp>
#include <SeQuant/core/utility/indices.hpp>
#include <SeQuant/core/utility/string.hpp>
#include <SeQuant/domain/mbpt/biorthogonalization.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/models/cc.hpp>
#include <SeQuant/domain/mbpt/op.hpp>
#include <SeQuant/domain/mbpt/rules/csv.hpp>
#include <SeQuant/domain/mbpt/rules/df.hpp>
#include <SeQuant/domain/mbpt/space_qns.hpp>
#include <SeQuant/domain/mbpt/utils.hpp>

#include <range/v3/view/tail.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>

using namespace sequant;

namespace {

// Phase O follow-up (2026-07-22, T2 deadlock root-cause fix):
// opt::eliminate_common_subexpressions()'s SubexpressionReplacer builds
// each OCCURRENCE's own reference from THAT occurrence's local
// canon_indices() (sorted only by index space) -- it does not carry over
// the actual bliss graph-isomorphism permutation found between the
// defining occurrence and a later, structurally-equal-but-axis-permuted
// occurrence. For a hoisted tensor with >=2 same-space indices where such
// a permutation exists, this can silently reference the SAME physical
// array with a TRANSPOSED axis order at different call sites -- confirmed
// by direct inspection of the generated T2 code: CSE17_i_i_i_i was
// defined and correctly referenced twice as ("i_3,i_4,i_1,i_2"), but
// referenced a THIRD time as ("i_3,i_4,i_2,i_1") -- the exact same four
// dummy-index tokens, just the last two positions swapped -- not a
// relabeling (a relabeling would use different token numbers), a genuine
// mislabeled axis order. This plausibly explains the observed MADNESS
// deadlock (a corrupted SparseShape/tile-dependency expectation from the
// mismatched axis order: a task gets scheduled expecting data in a tile
// combination that, under the real, untransposed sparsity pattern, is
// never produced).
//
// Fix: canonicalize every CSEn tensor's occurrences to its FIRST
// (defining) occurrence's annotation, whenever a later occurrence's
// annotation is a pure permutation of the exact same token set in a
// different order (a genuinely different token set, e.g. a differently-
// numbered contracted/free dummy index at a different call site, is left
// untouched -- that's normal, not a bug; verified separately that only
// token-set-identical, order-differing cases exist in the current output).
std::string fix_cse_axis_order(const std::string& code) {
  static const std::regex occ_re(R"(\b(CSE\d+[A-Za-z_μ̃Κ]*)\(\"([^\"]*)\"\))");

  std::unordered_map<std::string, std::string> canonical;
  for (auto it = std::sregex_iterator(code.begin(), code.end(), occ_re);
       it != std::sregex_iterator(); ++it) {
    const std::string& name = (*it)[1].str();
    if (!canonical.count(name)) canonical[name] = (*it)[2].str();
  }

  auto sorted_tokens = [](const std::string& ann) {
    std::vector<std::string> toks;
    std::stringstream ss(ann);
    std::string tok;
    while (std::getline(ss, tok, ',')) toks.push_back(tok);
    std::sort(toks.begin(), toks.end());
    return toks;
  };

  std::string result;
  std::size_t last_pos = 0;
  int fixes = 0;
  for (auto it = std::sregex_iterator(code.begin(), code.end(), occ_re);
       it != std::sregex_iterator(); ++it) {
    const std::smatch& m = *it;
    const std::string& name = m[1].str();
    const std::string& ann = m[2].str();
    result += code.substr(last_pos, m.position() - last_pos);
    auto found = canonical.find(name);
    if (found != canonical.end() && ann != found->second &&
        sorted_tokens(ann) == sorted_tokens(found->second)) {
      result += name + "(\"" + found->second + "\")";
      ++fixes;
    } else {
      result += m.str();
    }
    last_pos = static_cast<std::size_t>(m.position() + m.length());
  }
  result += code.substr(last_pos);

  if (fixes > 0) {
    std::cout << "  [fix_cse_axis_order] fixed " << fixes
              << " mislabeled CSE-tensor axis-order occurrence(s)\n";
  }
  return result;
}

// Phase O third follow-up (2026-07-22): fixes a SECOND, more severe
// generator bug found by bisecting T2's cross-term-CSE deadlock down to
// its minimal reproducing case (N=46 of 55 summands) and scanning the
// resulting code: a local C++ variable can get `+=`'d with a genuinely
// DIFFERENT outer rank than it was last written with, with no release in
// between -- e.g. `I_i_i_ap2_ap2` used as a rank-2 accumulator (its own
// final return value) throughout most of a function, but at one point
// `+=`'d with a rank-4 annotation. This is `eliminate_common_
// subexpressions`'s tree restructuring producing two genuinely
// different-rank SeQuant nodes that collide onto the same exported C++
// name (the export framework's own dedup, `TensorBlockLessThanComparator`,
// groups by label+slot/space signature, not full rank) -- confirmed via a
// dedicated scanner that this exact pattern appears ONLY in the failing
// generated code and never in working output. A rank-mismatched `+=` on a
// ToT array in a Release build (no shape assertion) plausibly corrupts
// internal SparseShape/tile-dependency bookkeeping, and because the
// corruption's actual failure mode depends on thread interleaving, the
// observed symptom (non-deterministic hang across identical reruns of the
// SAME generated code) is consistent with this being the root cause.
//
// Fix: scan for each `+=` whose outer rank differs from that name's
// last-tracked rank (from a PRIOR write with no release in between) --
// this is impossible for a legitimate accumulation, so treat every such
// span (from the colliding write to its closing release) as a genuinely
// SEPARATE local variable and rename it, isolating it completely from the
// name's other, correct uses. Verified end-to-end (5+ repeated real-data
// runs, no hangs, correct checksums matching the known-correct value)
// against a Python prototype before porting here.
std::string fix_rank_collision(const std::string& code) {
  std::vector<std::string> lines;
  {
    std::stringstream ss(code);
    std::string line;
    while (std::getline(ss, line)) lines.push_back(line);
  }

  static const std::regex decl_re(
      R"(^(\s*)(TA::TSpArrayD|TA::DistArray<[^;]+>)\s+([A-Za-z_][\w μ̃Κ]*);\s*$)");
  std::unordered_map<std::string, std::string> decl_type;
  std::unordered_map<std::string, std::size_t> decl_line_idx;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    std::smatch m;
    if (std::regex_match(lines[i], m, decl_re)) {
      decl_type[m[3].str()] = m[2].str();
      decl_line_idx[m[3].str()] = i;
    }
  }

  static const std::regex write_re(
      R"(^(\s*)([A-Za-z_][\w μ̃Κ]*)\(\"([^\"]*)\"\)\s*(\+?=))");
  static const std::regex release_re(
      R"(^(\s*)([A-Za-z_][\w μ̃Κ]*) = (.+\(\));\s*//\s*release\s*$)");

  auto outer_rank = [](const std::string& ann) -> int {
    std::string outer = ann.substr(0, ann.find(';'));
    if (outer.empty()) return 0;
    int n = 1;
    for (char c : outer) if (c == ',') ++n;
    return n;
  };

  std::unordered_map<std::string, int> current_rank;   // -1 == unwritten/released
  std::unordered_map<std::string, std::size_t> open_start;  // name -> collision start line
  struct Collision { std::size_t start, end; std::string name; };
  std::vector<Collision> collisions;

  for (std::size_t i = 0; i < lines.size(); ++i) {
    std::smatch rm;
    if (std::regex_match(lines[i], rm, release_re)) {
      const std::string& name = rm[2].str();
      auto oit = open_start.find(name);
      if (oit != open_start.end()) {
        collisions.push_back({oit->second, i, name});
        open_start.erase(oit);
      }
      current_rank[name] = -1;
      continue;
    }
    std::smatch wm;
    if (!std::regex_search(lines[i], wm, write_re)) continue;
    const std::string name = wm[2].str();
    const std::string ann = wm[3].str();
    const std::string op = wm[4].str();
    int rank = outer_rank(ann);
    if (open_start.count(name)) continue;  // already inside a collision span
    auto cit = current_rank.find(name);
    bool has_prior = cit != current_rank.end() && cit->second != -1;
    if (op == "+=" && has_prior && cit->second != rank) {
      open_start[name] = i;
    }
    current_rank[name] = rank;
  }

  for (const auto& [name, start] : open_start) {
    std::cout << "  [fix_rank_collision] WARNING: collision for " << name
              << " at statement " << (start + 1)
              << " never closes (no later release found) -- NOT fixed\n";
  }

  if (collisions.empty()) return code;

  std::vector<std::string> new_decls;
  int n = 0;
  for (const auto& c : collisions) {
    ++n;
    const std::string fresh = c.name + "_RANKFIX" + std::to_string(n);
    std::string type = "TA::TSpArrayD";
    if (auto it = decl_type.find(c.name); it != decl_type.end()) type = it->second;
    new_decls.push_back("  " + type + " " + fresh + ";");

    std::regex paren_re(R"(\b)" + c.name + R"(\()");
    for (std::size_t i = c.start; i <= c.end; ++i) {
      lines[i] = std::regex_replace(lines[i], paren_re, fresh + "(");
    }
    // Closing release line has no paren -- handle the bare-name form.
    std::regex bare_re(R"(\b)" + c.name + R"(\b(?!\())");
    lines[c.end] = std::regex_replace(lines[c.end], bare_re, fresh,
                                      std::regex_constants::format_first_only);
    // The collision's first write must be `=` (fresh var, never written).
    lines[c.start] =
        std::regex_replace(lines[c.start], std::regex(R"(\)\s*\+=)"), ") =",
                           std::regex_constants::format_first_only);
    std::cout << "  [fix_rank_collision] renamed span [" << (c.start + 1)
              << "," << (c.end + 1) << "]: " << c.name << " -> " << fresh
              << "\n";
  }

  std::size_t last_decl_line = 0;
  for (const auto& [name, idx] : decl_line_idx) last_decl_line = std::max(last_decl_line, idx);
  for (const auto& d : new_decls) lines[last_decl_line] += "\n" + d;

  std::string result;
  for (const auto& l : lines) { result += l; result += "\n"; }
  return result;
}

// Copied/adapted from mpqc4:src/mpqc/math/external/sequant/sequant.cpp's
// detail::make_sr_spaces() + load_convention().
std::shared_ptr<IndexSpaceRegistry> make_sr_spaces() {
  using namespace sequant::mbpt;
  auto isr = std::make_shared<IndexSpaceRegistry>();
  const auto spin_any = IndexSpace::QuantumNumbers{Spin::any};

  isr->add(L"i", 0b01, spin_any, is_vacuum_occupied, is_reference_occupied,
           is_hole)
      .add(L"a", 0b10, spin_any, is_particle)
      .add_union(L"p", {L"i", L"a"}, sequant::is_complete);
  mbpt::add_fermi_spin(*isr);
  mbpt::add_ao_spaces(isr, /*vbs=*/false, /*abs=*/false);
  mbpt::add_pao_spaces(isr);
  mbpt::add_df_spaces(isr);
  isr->physical_particle_attribute_mask(mask_v<Spin>);

  return isr;
}

void load_mpqc_sr_convention() {
  auto ctx = Context()
                 .set(Vacuum::SingleProduct)
                 .set(IndexSpaceMetric::Unit)
                 .set(SPBasis::Spinor)
                 .set(CanonicalizeOptions::default_options().copy_and_set(
                     CanonicalizationMethod::Complete))
                 .set(make_sr_spaces());
  set_default_context(ctx);
  mbpt::set_default_mbpt_context(
      mbpt::Context().set(mbpt::make_legacy_registry()));
}

// Copied verbatim from mpqc4:src/mpqc/chemistry/qc/lcao/cc/sequant.cpp.
ExprPtr tail_factor(ExprPtr const &expr) noexcept {
  if (expr->is<Tensor>())
    return expr->clone();
  else if (expr->is<Product>()) {
    auto scalar = expr->as<Product>().scalar();
    if (scalar == 1 && expr->size() == 2) {
      return expr->at(1);
    }
    auto facs = ranges::views::tail(*expr);
    return ex<Product>(Product{scalar, ranges::begin(facs), ranges::end(facs)});
  } else {
    auto summands = *expr | ranges::views::transform(
                                [](auto const &x) { return tail_factor(x); });
    return ex<Sum>(Sum{ranges::begin(summands), ranges::end(summands)});
  }
}

// Copied/adapted from mpqc4:src/mpqc/chemistry/qc/lcao/cc/sequant.cpp's
// make_cceqvec_csv_closedshell -- the biorthogonal-transform stage only.
// The SECOND tail_factor + density_fit + csv_transform + optimize stages
// (cck.ipp:1516-1540) are applied SEPARATELY below, matching
// generate_csv_closedshell()'s real control flow exactly.
std::vector<ExprPtr> make_cceqvec_csv_closedshell(std::size_t k) {
  using sequant::mbpt::CSV;
  using sequant::mbpt::set_scoped_default_mbpt_context;

  auto biorg = [](ExprPtr const &expr, Tensor const &S) -> ExprPtr {
    auto tf = tail_factor(expr);
    auto bt = mbpt::biorthogonal_transform(tf, external_indices(S));
    bt = S.clone() * bt;
    simplify(bt);
    return bt;
  };

  auto sequant_ctx = set_scoped_default_context(
      Context(get_default_context()).set(SPBasis::Spinfree));
  auto sequant_ctx_mbpt = set_scoped_default_mbpt_context(
      mbpt::Context({.csv = CSV::Yes, .op_registry_ptr = mbpt::make_legacy_registry()}));

  auto cc_r = mbpt::CC{k}.t();

  std::vector<ExprPtr> result;
  result.emplace_back(cc_r[0]);
  for (std::size_t r = 1; r <= k; ++r) {
    auto const S = cc_r[r]->front()->front().as<Tensor>();
    assert(S.label() == reserved::symm_label());
    result.emplace_back(biorg(cc_r[r], S));
  }
  return result;
}

// --- Phase 0 invariant check ---------------------------------------------

void collect_tensors(const ExprPtr &expr, std::vector<Tensor> &out) {
  if (expr->is<Tensor>()) {
    out.push_back(expr->as<Tensor>());
  } else {
    for (const ExprPtr &sub : *expr) collect_tensors(sub, out);
  }
}

// Returns false (and prints the offending tensor) if the invariant
// "has_proto_indices() on any index of a tensor <=> label in {C,t}" fails.
bool check_only_C_t_are_tot(const ExprPtr &expr) {
  std::vector<Tensor> tensors;
  collect_tensors(expr, tensors);
  bool ok = true;
  std::set<std::wstring> seen_bad;
  for (const Tensor &t : tensors) {
    bool any_proto = ranges::any_of(t.const_indices(), &Index::has_proto_indices);
    bool is_c_or_t = (t.label() == L"C" || t.label() == L"t");
    if (any_proto != is_c_or_t &&
        !seen_bad.contains(std::wstring(t.label()))) {
      std::wcerr << L"INVARIANT VIOLATION: tensor '" << t.label()
                 << L"' has_proto_indices=" << any_proto
                 << L" but label-is-C-or-t=" << is_c_or_t << L"\n";
      seen_bad.insert(std::wstring(t.label()));
      ok = false;
    }
  }
  return ok;
}

std::string family_sig(const Tensor &t) {
  std::ostringstream oss;
  bool first = true;
  for (const Index &idx : t.const_indices()) {
    if (!first) oss << ",";
    if (idx.has_proto_indices()) {
      oss << "[" << ranges::size(idx.proto_indices()) << "-proto]";
    } else {
      // narrow multi-byte space key crudely for a quick printable tag
      oss << "flat";
    }
    first = false;
  }
  return oss.str();
}

// Computes the enclosing equation's own true external/free index set
// directly from a (post-optimize()) Sum, WITHOUT relying on per-term local
// occurrence-count parity (get_unique_indices()'s existing Product-level
// cancellation logic gets this wrong whenever a genuine external index also
// happens to recur 3+ times within one term, e.g. the CSV/PNO occupied-pair
// "domain tag" -- see eval_expr.cpp's make_prod fix). Since all summands of
// a valid Sum MUST share the same external indices (that's what makes them
// addable as one equation), and SeQuant keeps a fixed equation's own named/
// external indices literal and identical across every summand (only internal
// Wick-contraction dummies get freshly renumbered per term), the robust,
// general signal is: an index is external iff it is present -- REGARDLESS of
// its local occurrence count -- in literally EVERY summand. Implemented as a
// straightforward intersection of each summand's own used-index set.
IndexSet sum_external_indices(const ExprPtr &e) {
  if (!e->is<Sum>()) return {};
  const auto &summands = e->as<Sum>().summands();
  if (summands.empty()) return {};

  IndexSet result = get_used_indices<IndexSet>(summands[0]);
  for (std::size_t i = 1; i < summands.size() && !result.empty(); ++i) {
    IndexSet const current = get_used_indices<IndexSet>(summands[i]);
    IndexSet intersected;
    for (auto &&ix : result)
      if (current.contains(ix)) intersected.emplace(ix);
    result = std::move(intersected);
  }
  return result;
}

void print_structural_summary(const ExprPtr &expr) {
  std::vector<Tensor> tensors;
  collect_tensors(expr, tensors);
  std::map<std::pair<std::wstring, std::string>, int> counts;
  for (const Tensor &t : tensors) {
    counts[{std::wstring(t.label()), family_sig(t)}]++;
  }
  for (const auto &[key, count] : counts) {
    std::wcout << L"  " << key.first << L"(" << key.second.c_str() << L") x"
               << count << L"\n";
  }
}

}  // namespace

int main() {
  load_mpqc_sr_convention();

  auto pao_uocc_space =
      get_default_context().index_space_registry()->retrieve(L"μ̃");
  auto dfbs_space =
      get_default_context().index_space_registry()->retrieve(L"Κ");

  std::cout << "Deriving closed-shell CSV CCSD residual equations "
               "(full cck.ipp-matching pipeline)...\n";
  auto residuals = make_cceqvec_csv_closedshell(2);

  for (std::size_t r = 1; r < residuals.size(); ++r) {
    ExprPtr e = residuals[r];
    std::cout << "\n=== Residual R" << r << " (post biorg, pre csv_transform) ===\n";
    std::cout << "Type: " << e->type_name();
    if (e->is<Sum>()) std::cout << ", " << e->as<Sum>().size() << " summands";
    std::cout << "\n";

    // cck.ipp:1518-1521
    e = tail_factor(e);            // second strip
    e = mbpt::density_fit(e, dfbs_space, L"g", L"g");  // df_ == true for this dataset
    e = mbpt::csv_transform(e, pao_uocc_space, L"C");  // default tensor_labels={f,g,S}

    // cck.ipp:1522-1539 (seq_opt_ == true for this dataset)
    flatten(e);
    OptimizeOptions opts;
    // SPTC_OPT_MEMSIZE (2026-07-29, cold-gap refactor attempt -- REJECTED, kept
    // as a gated experiment knob). Motivation: the OptFor::Flops cost model
    // (single_term.hpp flops_counter) has NO per-outer-cell/per-block task-
    // overhead term, so it prices the (μ̃,Κ)-both-outer DF half-transform
    // intermediate I_ap2_μ̃_Κ[i,i,μ̃,Κ;a] -- millions of tiny per-pair PNO cells
    // -- as cheap dense flops (it runs ~100x off peak at runtime; see
    // sequant-ta-repro docs/MPQC_EVALUATION.md §8). Hypothesis: OptFor::Memsize,
    // which penalizes the ~7e7-element intermediate, would pick an order that
    // never holds aux(Κ) and PAO(μ̃) open at once.
    // RESULT (measured, R2): it does the OPPOSITE -- Memsize emits MORE such
    // tiny-cell intermediates than Flops (3 vs 2; SPTC_NO_CSE=1 gives 4 -- so
    // cross-term CSE actually merges/reduces them). The (μ̃,Κ)-inner form is
    // structurally impossible anyway (inner ⇔ proto, and μ̃/Κ are non-proto/
    // global). The only setting that removes them is a large proto extent
    // (SPTC_PROTO_EXTENT=100 -> 0), but that swaps in a more-expensive
    // μ̃-family factorization that is slower with owning-ToT and numerically
    // divergent at cc-pVTZ. Conclusion: the (μ̃,Κ)-outer tiny-cell half-
    // transform is the flops-optimal factorization; no correctness-safe
    // generator knob avoids it. Knob kept (gated; OptFor::Flops stays default).
    opts.opt_for =
        std::getenv("SPTC_OPT_MEMSIZE") ? OptFor::Memsize : OptFor::Flops;
    opts.reorder = ReorderSum::Reorder;
    opts.is_volatile_leaf = [](Tensor const &t) { return t.label() == L"t"; };
    // FIX (2026-07-20, TA-vs-MPQC performance investigation): the values
    // below used to fall through to idx.space().approximate_size(), which
    // for i/mu-tilde/K is IndexSpace's own constructor DEFAULT of 10
    // (SeQuant/core/space.hpp) -- none of make_sr_spaces()'s add()/
    // add_pao_spaces()/add_df_spaces() calls pass a real size, unlike
    // MPQC's own real production path, which overwrites these via
    // populate_index_extents() (mpqc4:src/mpqc/math/external/sequant/
    // sequant.h) BEFORE deriving. Concretely this made optimize()'s
    // OptFor::Flops cost model see a mu-tilde^4-shaped intermediate as
    // ~10^4=10,000 "flops" when reality is 114^4~1.69e8 -- a ~16,900x
    // underestimate -- which made routing a contraction through a large
    // dense mu-tilde-family intermediate look nearly free, a very
    // plausible cause of whole_t2_residual OOMing past 62GB on real
    // ethane data (MPQC's own real g x g-family intermediates top out
    // around 625MB; nothing resembling a dense mu-tilde^4 object ever
    // appears in its real trace). Real values below are read directly off
    // this exact ethane run's own registry dump
    // (mpqc4/traces/checksum-run/ethane-checksum-v2.log:462-474) --
    // NOTE these are the real ACTIVE-space counts (i=7), not the raw/
    // padded COO array shape (9, which includes frozen-core rows with an
    // all-zero PNO domain) -- the registry dump is what MPQC's own
    // optimize() call actually sees, so it's the correct number to
    // replicate here.
    // Perf-parity Phase B: SPTC_PROTO_EXTENT / SPTC_NREPLAY override the
    // optimizer's proto (PNO) extent and replay count to test whether the
    // giant μ̃³ / μ̃Κ DF intermediates the trace found are an artifact of
    // the extent model or an under-searched contraction order.
    std::size_t proto_ext = 45;
    if (const char *v = std::getenv("SPTC_PROTO_EXTENT")) proto_ext = std::atoi(v);
    std::size_t nreplay = 10;
    if (const char *v = std::getenv("SPTC_NREPLAY")) nreplay = std::atoi(v);
    opts.idx_to_extent = [proto_ext](Index const &idx) -> std::size_t {
      if (idx.has_proto_indices()) return proto_ext;
      const std::wstring &key = idx.space().base_key();
      if (key == L"i") return 7;     // occupied (active space only)
      if (key == L"μ̃")    // mu-tilde (CSV/PAO-restricted basis)
        return 114;
      if (key == L"Κ") return 282;  // DF/RI auxiliary basis
      return idx.space().approximate_size();
    };
    opts.n_replay = nreplay;
    e = optimize(e, opts);

    std::cout << "=== Residual R" << r << " (post full pipeline) ===\n";
    std::cout << "Type: " << e->type_name();
    if (e->is<Sum>()) std::cout << ", " << e->as<Sum>().size() << " summands";
    std::cout << "\n";

    bool ok = check_only_C_t_are_tot(e);
    std::cout << "Invariant (only C/t carry proto_indices): "
              << (ok ? "HOLDS" : "VIOLATED") << "\n";
    std::cout << "Structural summary (label(index-family-sig) x count):\n";
    print_structural_summary(e);

    // The enclosing equation's own true external/free index set (e.g. {i;a}
    // for R1, {i,j;a,b} for R2), derived automatically from cross-summand
    // commonality -- see sum_external_indices()'s doc comment. This is
    // threaded through to_export_tree()/binarize() as the authoritative
    // survival signal for eval_expr.cpp's make_prod fix: a "domain tag"
    // index (e.g. the CSV/PNO occupied-pair index reused across several
    // C-transform tensors and the T amplitude within one term) is exactly
    // one of THESE indices, and must survive a term's own binarization
    // regardless of how many times it happens to recur locally -- while a
    // term-local artifact index that merely LOOKS the same (same local
    // occurrence count) but is NOT one of these must not be force-preserved.
    IndexSet const eq_external = sum_external_indices(e);
    std::wcout << L"Equation R" << r << L"'s own external indices ("
               << eq_external.size() << L"): ";
    for (auto &&ix : eq_external) std::wcout << ix.label() << L" ";
    std::wcout << L"\n";

    // Phase O (2026-07-22, performance-parity investigation): optimize()
    // only CSEs WITHIN one top-level Sum term (core/optimize/optimize.cpp
    // processes each summand independently, explicitly parallelized with
    // no cross-term shared state) -- it never detects that a structurally
    // identical subexpression recurs across DIFFERENT top-level terms
    // under a different dummy-index gensym. Confirmed empirically (see
    // ~/.claude/jobs/*/tmp/dedup_check.py): 65-72% of the statements this
    // pipeline previously generated were exact duplicates (modulo
    // consistent dummy-index renaming) of a computation already done
    // elsewhere in the same file. Real MPQC's own runtime evaluator avoids
    // this via a cross-term CacheManager (cck.ipp:1679-1692,
    // cache_manager(nodes, L"t", min_repeats=2)); this block is SeQuant's
    // own static equivalent (opt::eliminate_common_subexpressions(),
    // previously unused anywhere in the tree -- confirmed via
    // `grep -rl eliminate_common_subexpressions`). AcceptAllPredicate
    // (the default filter) is safe here because our benchmark evaluates
    // the whole residual ONCE against fixed leaf data -- unlike MPQC's
    // real iterative solve, there's no "amplitude changes every
    // iteration" staleness concern to gate on.
    //
    // Build the per-summand export forest -- same to_export_tree() call
    // and same `summands`/`eq_external` the pre-existing per-summand
    // cross-check loop below already uses -- then run cross-term CSE on
    // it before exporting, instead of exporting the whole-Sum tree
    // directly (which never shared anything across summands to begin
    // with).
    // SAFETY (2026-07-22): T1's cross-term-CSE output verified correct
    // (checksum matches the known-correct value exactly). T2's, however,
    // triggers a real, NON-DETERMINISTIC MADNESS/TiledArray deadlock
    // ("Hung queue?" / "0 of 2401 tiles set" internally, or an external
    // wall-clock hang) -- confirmed via bisection (SPTC_CSE_BISECT_N
    // below) that it reproduces reliably at N>=46 of T2's 55 summands and
    // never at N<=45, and via repeated identical reruns that the SAME
    // generated code sometimes completes correctly and sometimes hangs
    // (ruling out a simple deterministic logic bug in favor of a genuine
    // runtime race).
    //
    // TWO real, distinct bugs found via direct inspection of the generated
    // code (not guessing), both upstream of our own generator/export
    // pipeline:
    // 1. eliminate_common_subexpressions()'s SubexpressionReplacer built
    //    each occurrence's reference from THAT occurrence's own local
    //    canon_indices() (sorted only by index space), without correcting
    //    for a same-space-index permutation a bliss graph-isomorphism can
    //    equate as "the same" subexpression -- confirmed in T2's output
    //    (CSE17_i_i_i_i referenced with a transposed axis order at one
    //    call site: same 4 dummy tokens, last two positions swapped).
    //    Fixed via fix_cse_axis_order() above (canonicalizes any CSEn
    //    tensor's occurrences to its defining occurrence's annotation
    //    whenever a later one is a pure permutation of the same token
    //    set). Necessary, but NOT sufficient on its own to unblock T2.
    // 2. THE LIKELY ROOT CAUSE of the actual deadlock: a C++-variable-NAME
    //    COLLISION ACROSS GENUINELY DIFFERENT RANKS. Found by bisecting to
    //    the minimal N=46 case and scanning its generated code: the local
    //    variable `I_i_i_ap2_ap2` is used as a rank-2-outer ToT array
    //    (annotation "i_1,i_2;a_1,a_2") throughout MOST of the function
    //    (including as the function's own return value), but at ONE point
    //    gets `+=`'d with a RANK-4-outer annotation
    //    ("i_1,i_2,i_3,i_4;a_3,a_4") with NO release/reset in between --
    //    i.e. the exact same C++ object accumulates two shape-incompatible
    //    einsum results. Confirmed via a dedicated scanner
    //    (check_rank_collision.py) that this exact pattern is present ONLY
    //    in the N>=46 generated code and absent from N<=45 and from T1's
    //    (working) CSE output -- a precise, reproducible correlation with
    //    the bisected failure boundary. Root cause: the export framework's
    //    C++-variable naming/dedup scheme groups intermediates by a family
    //    SIGNATURE (label + slot/space shape) that `eliminate_common_
    //    subexpressions`'s tree restructuring can apparently violate --
    //    two genuinely different-rank SeQuant nodes ending up mapped to
    //    the same exported name without an intervening reset. This is
    //    consistent with the observed NON-DETERMINISM: a rank-mismatched
    //    `+=` on a ToT DistArray in a Release build (no shape assertion)
    //    plausibly corrupts SparseShape/tile-dependency bookkeeping in a
    //    way whose exact failure mode (silent-but-wrong vs. deadlock)
    //    depends on thread interleaving, not just the code itself.
    //    **FIXED** (2026-07-22, third follow-up) via `fix_rank_collision()`
    //    below -- detects any `+=` whose outer rank differs from that
    //    name's last-tracked rank (impossible for a legitimate
    //    accumulation) and isolates the whole colliding span into a
    //    freshly-named, dedicated variable. Verified via 5+ repeated
    //    real-data runs: zero hangs, checksums matching the known-correct
    //    value every time -- CSE is now enabled for BOTH T1 and T2.
    //
    // Bisection knob (kept as a reusable diagnostic tool): SPTC_CSE_BISECT_N,
    // if set and r==2, runs cross-term CSE on only the FIRST N of T2's 55
    // summands (the rest export individually, un-deduped but still
    // correct) -- this is exactly how bug 2 above was isolated to its
    // minimal N=46 reproducing case, which led directly to the fix above.
    // SPTC_NO_CSE=1 disables cross-term CSE (performance-parity Phase B:
    // the per-op trace showed cross-term CSE creates giant merged
    // intermediates, e.g. a 70M-nnz i,i,μ̃,Κ;a node = ~1s of T2, that
    // MPQC's per-summand trees never form — test per-summand-only codegen).
    bool use_cross_term_cse = !(std::getenv("SPTC_NO_CSE") &&
                                std::atoi(std::getenv("SPTC_NO_CSE")) != 0);
    int cse_bisect_n = -1;
    if (r == 2) {
      if (const char* v = std::getenv("SPTC_CSE_BISECT_N")) {
        cse_bisect_n = std::atoi(v);
        use_cross_term_cse = true;
      }
    }
    container::svector<ExportNode<ExportExpr>> forest;
    if (use_cross_term_cse && e->is<Sum>()) {
      const auto &cse_summands = e->as<Sum>().summands();
      std::size_t n_cse = cse_bisect_n >= 0
                               ? std::min<std::size_t>(cse_bisect_n, cse_summands.size())
                               : cse_summands.size();
      if (cse_bisect_n >= 0) {
        std::cout << "  [bisect] CSE-ing first " << n_cse << "/"
                  << cse_summands.size() << " summands\n";
      }
      container::svector<ExportNode<ExportExpr>> cse_forest;
      cse_forest.reserve(n_cse);
      for (std::size_t i = 0; i < n_cse; ++i) {
        cse_forest.push_back(
            to_export_tree(cse_summands[i], /*retain_braket=*/false, eq_external));
      }
      auto expr_to_tree = [&](const auto &x) -> ExportNode<ExportExpr> {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>,
                                     ExprPtr>) {
          return to_export_tree<ExportExpr>(x, /*retain_braket=*/false,
                                            eq_external);
        } else {
          return to_export_tree<ExportExpr>(x, /*retain_braket=*/false);
        }
      };
      // NOTE: eliminate_common_subexpressions() is single-pass by design --
      // it only detects duplicates present in the ORIGINAL trees, so a
      // duplicate that only becomes apparent after an earlier round's
      // hoisting isn't caught in one call. Tried iterating this to a fixed
      // point: T2's forest converges cleanly (55->121 trees over 5 passes,
      // stable after), but T1's grows by a steady +2 trees/pass with no
      // sign of convergence even after 50 passes -- behavior this
      // single-call API wasn't designed/verified for, and not something
      // this investigation chased down further. Sticking to the single,
      // well-understood call the approved plan scoped.
      sequant::opt::eliminate_common_subexpressions(cse_forest, expr_to_tree);
      forest = std::move(cse_forest);
      // Remaining summands (bisection only): export individually,
      // un-deduped -- still correct, just not shared.
      for (std::size_t i = n_cse; i < cse_summands.size(); ++i) {
        forest.push_back(
            to_export_tree(cse_summands[i], /*retain_braket=*/false, eq_external));
      }
    } else {
      forest.push_back(to_export_tree(e, /*retain_braket=*/false, eq_external));
    }
    TiledArrayGenerator generator;
    TiledArrayGeneratorContext ctx;
    std::string fn_name = "whole_t" + std::to_string(r) + "_residual";
    try {
      export_group(ExpressionGroup<ExportExpr>{std::move(forest), fn_name},
                  generator, ctx);
      std::string code = generator.get_generated_code();
      if (use_cross_term_cse) {
        code = fix_cse_axis_order(code);
        code = fix_rank_collision(code);
      }
      std::cout << "Generated " << code.size() << " chars of C++.\n";
      std::string path =
          "/tmp/claude-ta-generator-test/generated_R" + std::to_string(r) + ".cpp";
      std::ofstream out(path);
      out << "#include <tiledarray.h>\n#include <TiledArray/expressions/"
             "einsum.h>\n#include <cmath>\n\n"
          << code;
      std::cout << "wrote " << path << "\n";
      std::cout << "=== Leaf manifest (R" << r << ") ===\n"
                << generator.leaf_manifest_report();
    } catch (const std::exception &ex) {
      std::cout << "EXCEPTION: " << ex.what() << "\n";
    }

    // --- Per-summand export (term-by-term numeric cross-check) ----------
    // Phase 5 tier-3 comparison methodology (twinkly-dazzling-shamir.md):
    // export EACH top-level summand of the post-optimize Sum as its own
    // small named function so its checksum can be computed independently
    // and matched against the corresponding real MPQC trace row, instead
    // of only comparing the whole-residual sum.
    if (e->is<Sum>()) {
      const auto &summands = e->as<Sum>().summands();
      std::string manifest_path =
          "/tmp/claude-ta-generator-test/r" + std::to_string(r) + "_terms.tsv";
      std::ofstream term_manifest(manifest_path);
      std::cout << "\n=== Exporting " << summands.size() << " individual R"
                << r << " summands ===\n";
      for (std::size_t t = 0; t < summands.size(); ++t) {
        const ExprPtr &term = summands[t];
        std::string fn_name =
            "t" + std::to_string(r) + "_term" + std::to_string(t);
        std::string text = toUtf8(io::serialization::to_string(term));
        // TSV-safe: strip tabs/newlines from the printed expression.
        for (char &c : text)
          if (c == '\t' || c == '\n' || c == '\r') c = ' ';

        term_manifest << fn_name << "\t" << text << "\t";
        try {
          auto term_tree =
              to_export_tree(term, /*retain_braket=*/false, eq_external);
          TiledArrayGenerator term_gen;
          TiledArrayGeneratorContext term_ctx;
          export_group(
              ExpressionGroup<ExportExpr>{std::move(term_tree), fn_name},
              term_gen, term_ctx);
          std::string code = term_gen.get_generated_code();
          std::string path = "/tmp/claude-ta-generator-test/generated_r" +
                             std::to_string(r) + "_term" + std::to_string(t) +
                             ".cpp";
          std::ofstream out(path);
          out << "#include <tiledarray.h>\n#include <TiledArray/expressions/"
                 "einsum.h>\n#include <cmath>\n\n"
              << code;
          // Compact leaf manifest, one term's params joined by ';':
          //   name:label:outer1|outer2:inner1|inner2:ToT|flat
          const auto &leaves = term_gen.leaf_manifest();
          for (std::size_t li = 0; li < leaves.size(); ++li) {
            if (li) term_manifest << ";";
            const auto &lf = leaves[li];
            term_manifest << lf.name << ":" << lf.label << ":";
            for (std::size_t k = 0; k < lf.outer_families.size(); ++k) {
              if (k) term_manifest << "|";
              term_manifest << lf.outer_families[k];
            }
            term_manifest << ":";
            for (std::size_t k = 0; k < lf.inner_families.size(); ++k) {
              if (k) term_manifest << "|";
              term_manifest << lf.inner_families[k];
            }
            term_manifest << ":" << (lf.is_tot ? "ToT" : "flat");
          }
          term_manifest << "\n";
        } catch (const std::exception &ex) {
          std::string reason = ex.what();
          for (char &c : reason)
            if (c == '\t' || c == '\n' || c == '\r') c = ' ';
          term_manifest << "EXCEPTION:" << reason << "\n";
        }
      }
      std::cout << "wrote " << manifest_path << "\n";
    }
  }

  return 0;
}
