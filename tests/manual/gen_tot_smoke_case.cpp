// Emits the actual generated C++ for the Phase 3 ground-truth smoke test:
// R(i,x) = sum_a' T(i;a') * C(i,x;a')  -- ToT x ToT -> flat de-nest pattern.
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

  auto T = ex<Tensor>(L"T", bra{i1}, ket{a1p});
  auto C = ex<Tensor>(L"C", bra{a1p}, ket{x1});
  auto prod = ex<Product>(Product{1, {T, C}});

  // NOTE: to_export_tree's automatic free-index inference treats i1 as
  // CONTRACTED here (it's shared between T's own bra and C's proto-
  // extracted outer identity) -- confirmed empirically (an earlier run of
  // this generator produced a rank-1 result missing i1 entirely). That's
  // Product's own algebra, not a generator bug, but it means the natural
  // "both i and x survive" back-transform shape (matching a real R1(i,a)-
  // style residual) needs the result tensor specified EXPLICITLY via
  // compute(), the same workaround used for the mixed-case unit test.
  Tensor result(L"R", bra{i1}, ket{x1});
  TiledArrayGenerator generator;
  TiledArrayGeneratorContext gctx;
  generator.begin_export(gctx);
  generator.begin_named_section("compute_R", gctx);
  UsageSet terminal;
  terminal = Usage::Terminal;
  UsageSet intermediate;
  intermediate = Usage::Intermediate;
  generator.declare(T->as<Tensor>(), terminal, gctx);
  generator.declare(C->as<Tensor>(), terminal, gctx);
  generator.declare(result, intermediate, gctx);
  generator.compute(*prod, result, gctx);
  generator.persist(result, gctx);
  generator.end_named_section("compute_R", gctx);
  generator.end_export(gctx);
  std::string code = generator.get_generated_code();

  std::cout << code;
  std::ofstream out("/tmp/claude-ta-generator-test/tot_smoke/generated_case.h");
  out << code;
  std::cerr << "wrote generated_case.h (" << code.size() << " chars)\n";
  return 0;
}
