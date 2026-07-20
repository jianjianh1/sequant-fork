// Counter-example probe: does a domain-tag index that gets FULLY GATHERED
// well before the top of the tree (with unrelated combinations happening
// afterward) still need to survive to the final result? Mirrors the EXACT
// shape of generated_r1_term0.cpp's own i_2 (occupied-pair domain tag):
//   X1(i2;a2) * X2(i2;a2) -- contract a2, i2 survives (2 of 3 occurrences)
//   (X1*X2) * X3(i2,i1)   -- i2's 3rd (LAST) occurrence folds in HERE;
//                            i1's 1st occurrence appears
//   ((X1*X2)*X3) * X4(i1;a1) -- i1's 2nd occurrence folds in; i2 is NOT
//                            touched by X4 at all, yet MUST still survive to
//                            the final result (matches I_i_ap1("i_1,i_2;a_1")
//                            in the real generated code).
//
// This has the IDENTICAL raw-occurrence-count shape as the orphan-index
// repro (test_orphan_index_survives_too_long.cpp): an index (i2) reaches
// its whole-term total (3) partway through the tree, then MORE combining
// happens afterward that doesn't touch it -- yet here it MUST keep
// surviving (real domain tag), whereas in the orphan-index repro it must
// NOT (the whole point of that repro). Used to check whether a "local total
// vs whole-term total" comparison in make_prod can soundly distinguish the
// two cases at all.
#include <SeQuant/core/context.hpp>
#include <SeQuant/core/export/export.hpp>
#include <SeQuant/core/export/tiledarray_generator.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/core/io/shorthands.hpp>
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

  Index i1(occ, 1);
  Index i2(occ, 2);
  Index a1(virt, 1);
  Index a2(virt, 2);

  auto X1 = ex<Tensor>(L"X1", bra{i2}, ket{a2});
  auto X2 = ex<Tensor>(L"X2", bra{i2}, ket{a2});
  auto X3 = ex<Tensor>(L"X3", bra{i2}, ket{i1});
  auto X4 = ex<Tensor>(L"X4", bra{i1}, ket{a1});

  auto X1X2 = ex<Product>(Product{1, {X1, X2}, Product::Flatten::No});
  auto X1X2X3 = ex<Product>(Product{1, {X1X2, X3}, Product::Flatten::No});
  auto whole = ex<Product>(Product{1, {X1X2X3, X4}, Product::Flatten::No});

  std::cout << "Hand-parenthesized expression: " << toUtf8(whole->to_latex())
            << "\n";
  std::cout << "i2's whole-term occurrence count is 3 (X1,X2,X3), fully "
               "gathered by the (X1*X2)*X3 step -- X4 (combined afterward) "
               "does not touch i2 at all. i2 MUST still survive to the "
               "final result (it's the term's genuine domain-tag external "
               "index, exactly like the real i_2 in "
               "generated_r1_term0.cpp's I_i_ap1(\"i_1,i_2;a_1\")).\n";

  // The enclosing equation's TRUE external/free index set includes i2 (the
  // domain tag, which is genuinely one of this equation's own external
  // indices, just reused as a restriction tag on X1/X2/X3) as well as i1
  // and a1 (this term's other genuinely-surviving legs). Supplying this
  // explicitly is what lets make_prod's fix (eval_expr.cpp) tell this case
  // apart from the structurally-identical term-local-artifact case in
  // test_orphan_index_survives_too_long.cpp.
  IndexSet const external{i1, i2, a1};

  std::string code =
      export_and_print(whole, "((X1*X2)*X3)*X4", external);

  bool has_i2 = code.find("i_2") != std::string::npos;
  std::cout << "\n=== VERDICT ===\n";
  std::cout << "i_2 appears anywhere in generated code: " << has_i2
            << "  (expected: true -- i2 is a genuine domain tag)\n";
  if (!code.empty()) {
    // Find the LAST statement's own target annotation (heuristic: last
    // '("..."' occurrence terminated by ')').
    auto pos = code.rfind("(\"");
    std::cout << "Tail of generated code (last statement(s)):\n"
              << code.substr(code.size() > 400 ? code.size() - 400 : 0)
              << "\n";
  }
  return 0;
}
