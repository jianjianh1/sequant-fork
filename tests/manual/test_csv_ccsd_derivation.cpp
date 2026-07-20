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

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace sequant;

namespace {

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
    opts.opt_for = OptFor::Flops;
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
    opts.idx_to_extent = [](Index const &idx) -> std::size_t {
      if (idx.has_proto_indices()) {
        // Real average per-pair PNO domain size, measured directly from
        // this ethane dataset's t_i_1_i_2_a_1_a_2.txt (49 real occupied
        // pairs, contiguous-range convention matching ta_builder.h's
        // build_tot_array()) -- was hardcoded to a generic guess of 30.
        return 45;
      }
      const std::wstring &key = idx.space().base_key();
      if (key == L"i") return 7;     // occupied (active space only)
      if (key == L"μ̃")    // mu-tilde (CSV/PAO-restricted basis)
        return 114;
      if (key == L"Κ") return 282;  // DF/RI auxiliary basis
      return idx.space().approximate_size();
    };
    opts.n_replay = 10;
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

    auto tree = to_export_tree(e, /*retain_braket=*/false, eq_external);
    TiledArrayGenerator generator;
    TiledArrayGeneratorContext ctx;
    std::string fn_name = "whole_t" + std::to_string(r) + "_residual";
    try {
      export_group(ExpressionGroup<ExportExpr>{std::move(tree), fn_name},
                  generator, ctx);
      std::string code = generator.get_generated_code();
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
