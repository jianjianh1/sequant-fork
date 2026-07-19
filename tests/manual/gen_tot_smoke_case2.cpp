// Case 2 (Hadamard pass-through, flat x ToT -> ToT): R2(i;a') = sum_x
// g(i,x) * C(i,x;a'). Companion to gen_tot_smoke_case.cpp's de-nest case --
// same C data/formula, exercising the OTHER major codegen path (plain,
// non-DeNest TA::einsum preserving a genuine ToT result).
#include <SeQuant/core/context.hpp>
#include <SeQuant/core/export/export.hpp>
#include <SeQuant/core/export/tiledarray_generator.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/space_qns.hpp>

#include <fstream>
#include <iostream>

using namespace sequant;

int main() {
  using namespace sequant::mbpt;
  auto isr = std::make_shared<IndexSpaceRegistry>();
  const auto spin_any = IndexSpace::QuantumNumbers{Spin::any};
  isr->add(L"i", 0b01, spin_any, is_vacuum_occupied, is_reference_occupied,
           is_hole)
      .add(L"a", 0b10, spin_any, is_particle)
      .add_union(L"p", {L"i", L"a"}, sequant::is_complete);
  mbpt::add_fermi_spin(*isr);
  isr->physical_particle_attribute_mask(mask_v<Spin>);

  auto ctx = sequant::Context()
                 .set(Vacuum::SingleProduct)
                 .set(IndexSpaceMetric::Unit)
                 .set(SPBasis::Spinor)
                 .set(isr);
  set_default_context(ctx);

  IndexSpace occ = isr->retrieve(L"i");
  IndexSpace virt = isr->retrieve(L"a");

  Index i1(occ, 1);
  Index x1(virt, 1);
  Index a1p(virt, 2, container::vector<Index>{i1});

  auto g = ex<Tensor>(L"g", bra{i1}, ket{x1});
  auto C = ex<Tensor>(L"C", bra{a1p}, ket{x1});
  auto prod = ex<Product>(Product{1, {g, C}});

  // Explicit result R2(i1;a1p): i1 survives as genuine outer (same reason
  // as the de-nest case -- to_export_tree's auto free-index inference
  // would otherwise contract i1 away since it's shared via C's proto
  // extraction).
  Tensor result(L"R2", bra{i1}, ket{a1p});
  TiledArrayGenerator generator;
  TiledArrayGeneratorContext gctx;
  generator.begin_export(gctx);
  generator.begin_named_section("compute_R2", gctx);
  UsageSet terminal;
  terminal = Usage::Terminal;
  UsageSet intermediate;
  intermediate = Usage::Intermediate;
  generator.declare(g->as<Tensor>(), terminal, gctx);
  generator.declare(C->as<Tensor>(), terminal, gctx);
  generator.declare(result, intermediate, gctx);
  generator.compute(*prod, result, gctx);
  generator.persist(result, gctx);
  generator.end_named_section("compute_R2", gctx);
  generator.end_export(gctx);
  std::string code = generator.get_generated_code();

  std::cout << code;
  std::ofstream out(
      "/tmp/claude-ta-generator-test/tot_smoke2/generated_case2.h");
  out << code;
  std::cerr << "wrote generated_case2.h (" << code.size() << " chars)\n";
  return 0;
}
