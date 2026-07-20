// Minimal, standalone, COMPILED repro for the suspected SeQuant core bug:
// a "domain tag" index shared by 3+ tensor factors in a Product (occurs more
// than the standard Wick-contraction 2 times) gets silently dropped from an
// intermediate's own exposed index set somewhere inside optimize()+binarize(),
// even though it must survive all the way to the top (it's a genuine free/
// external index of the whole term, e.g. the CSV-CCSD occupied-pair index that
// tags several C-transform tensors AND the T2 amplitude).
//
// Deliberately mirrors the REAL pipeline used by
// tests/manual/test_csv_ccsd_derivation.cpp: build a flat Product, run it
// through sequant::optimize() (DP-based flops-minimizing binarization, same
// options family), then sequant::to_export_tree() (which calls
// binarize(expr, /*uncontract=*/{}, ...) -- note the EMPTY uncontract, the
// same bare-ExprPtr call the real generator uses), then export via
// TiledArrayGenerator and print the generated C++ so the exact intermediate
// where the domain-tag index vanishes is directly visible.
#include <SeQuant/core/context.hpp>
#include <SeQuant/core/export/export.hpp>
#include <SeQuant/core/export/tiledarray_generator.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/space_qns.hpp>

#include <iostream>
#include <string>

using namespace sequant;

namespace {

std::shared_ptr<IndexSpaceRegistry> make_spaces() {
  using namespace sequant::mbpt;
  auto isr = std::make_shared<IndexSpaceRegistry>();
  const auto spin_any = IndexSpace::QuantumNumbers{Spin::any};
  isr->add(L"i", 0b01, spin_any, is_vacuum_occupied, is_reference_occupied,
           is_hole)
      .add(L"a", 0b10, spin_any, is_particle)
      .add_union(L"p", {L"i", L"a"}, sequant::is_complete);
  mbpt::add_fermi_spin(*isr);
  isr->physical_particle_attribute_mask(mask_v<Spin>);
  return isr;
}

void load_convention() {
  auto ctx = Context()
                 .set(Vacuum::SingleProduct)
                 .set(IndexSpaceMetric::Unit)
                 .set(SPBasis::Spinor)
                 .set(make_spaces());
  set_default_context(ctx);
  mbpt::set_default_mbpt_context(
      mbpt::Context().set(mbpt::make_legacy_registry()));
}

std::string export_and_print(const ExprPtr &e, const std::string &label,
                              IndexSet const &external = {}) {
  auto tree = to_export_tree(e, /*retain_braket=*/false, external);
  TiledArrayGenerator generator;
  TiledArrayGeneratorContext ctx;
  try {
    export_expression(std::move(tree), generator, ctx);
    std::string code = generator.get_generated_code();
    std::cout << "\n=== " << label << " : generated code ===\n" << code << "\n";
    return code;
  } catch (const std::exception &ex) {
    std::cout << "\n=== " << label << " : EXCEPTION: " << ex.what() << " ===\n";
    return "";
  }
}

}  // namespace

int main() {
  load_convention();
  auto isr = get_default_context().index_space_registry();
  IndexSpace occ = isr->retrieve(L"i");
  IndexSpace virt = isr->retrieve(L"a");

  // Domain-tag index pair (mimics the occupied-pair PNO-domain restriction):
  // shared literally by T, C1, AND C2 -- three occurrences, NOT the standard
  // two.  a1 is contracted between T and C1 (2 occurrences -- a normal Wick
  // dummy).  a2 is contracted between T and C2 (2 occurrences).  b1, b2 are
  // genuine free/external indices (1 occurrence each).
  Index i1(occ, 1);
  Index i2(occ, 2);
  Index a1(virt, 1);
  Index a2(virt, 2);
  Index b1(virt, 3);
  Index b2(virt, 4);

  auto T = ex<Tensor>(L"T", bra{i1, i2}, ket{a1, a2});
  auto C1 = ex<Tensor>(L"C1", bra{i1, i2}, ket{a1, b1});
  auto C2 = ex<Tensor>(L"C2", bra{i1, i2}, ket{a2, b2});

  // The whole term's TRUE free/external indices are {i1, i2, b1, b2} -- i1,i2
  // must survive to the top exactly like b1,b2 do, even though (unlike b1/b2)
  // they occur 3 times, not once.
  auto prod = ex<Product>(Product{1, {T, C1, C2}});

  std::cout << "Product built: " << toUtf8(prod->to_latex()) << "\n";

  // The whole term's TRUE free/external indices -- supplied explicitly here
  // (as the fix for the make_prod domain-tag-vs-artifact ambiguity requires:
  // binarize() cannot soundly derive this from local occurrence counts
  // alone, see eval_expr.cpp's make_prod). i1,i2 recur 3x each (domain tag);
  // b1,b2 occur once each (plain externals) -- ALL FOUR must survive.
  IndexSet const external{i1, i2, b1, b2};

  // --- Step 1: export WITHOUT optimize() first (flat 3-factor left-to-right
  // fold).  Included for contrast/diagnosis only.
  export_and_print(prod, "flat (no optimize())", external);

  // --- Step 2: run through the REAL pipeline's optimize() call (same option
  // family as test_csv_ccsd_derivation.cpp: Flops metric, volatile-leaf
  // weighting disabled here since there's no 't' label ambiguity, subnet_cse
  // left at its default Disable).
  OptimizeOptions opts;
  opts.opt_for = OptFor::Flops;
  opts.reorder = ReorderSum::NoReorder;  // single term, no Sum to reorder
  opts.idx_to_extent = [](Index const &idx) -> std::size_t {
    return idx.space().approximate_size();
  };
  ExprPtr optimized = optimize(prod, opts);
  std::cout << "\nOptimized (parenthesized) expression: "
            << toUtf8(optimized->to_latex()) << "\n";

  std::string code = export_and_print(optimized, "post-optimize()", external);

  bool has_i1 = code.find("i_1") != std::string::npos;
  bool has_i2 = code.find("i_2") != std::string::npos;
  std::cout << "\n=== VERDICT ===\n";
  std::cout << "i_1 appears anywhere in generated code: " << has_i1 << "\n";
  std::cout << "i_2 appears anywhere in generated code: " << has_i2 << "\n";

  // Stronger check: the domain tag must appear in the TOP-level (last)
  // generated statement's OWN result annotation, not just somewhere in an
  // intermediate that never reaches the top.
  auto last_stmt_pos = code.rfind("R_");  // heuristic: exported result name
  std::cout << "(see full generated code above to confirm whether i_1/i_2 "
               "survive into the FINAL statement's own result annotation, "
               "vs. only appearing in early intermediates and then vanishing)"
            << "\n";

  return 0;
}
