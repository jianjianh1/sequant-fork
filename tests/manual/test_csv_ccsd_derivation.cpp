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
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/core/rational.hpp>
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
    opts.idx_to_extent = [](Index const &idx) -> std::size_t {
      if (idx.has_proto_indices()) return 30;  // plausible avg PNO count
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

    auto tree = to_export_tree(e, /*retain_braket=*/false);
    TiledArrayGenerator generator;
    TiledArrayGeneratorContext ctx;
    try {
      export_expression(std::move(tree), generator, ctx);
      std::string code = generator.get_generated_code();
      std::cout << "Generated " << code.size() << " chars of C++.\n";
      std::string path =
          "/tmp/claude-ta-generator-test/generated_R" + std::to_string(r) + ".cpp";
      std::ofstream out(path);
      out << "#include <tiledarray.h>\n#include <TiledArray/expressions/"
             "einsum.h>\n#include <cmath>\n\n"
          << code;
      std::cout << "wrote " << path << "\n";
    } catch (const std::exception &ex) {
      std::cout << "EXCEPTION: " << ex.what() << "\n";
    }
  }

  return 0;
}
