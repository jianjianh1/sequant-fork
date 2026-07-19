#include <SeQuant/core/context.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/space_qns.hpp>

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

  // NOTE: Index::make_tmp_index() mints ordinals >= min_tmp_index(), which
  // Index::drop_proto_indices() (going through the string-label ctor
  // internally) explicitly rejects (check_nonreserved(), index.hpp:918-923)
  // -- confirmed the real crash cause. Use plain low-ordinal indices
  // instead, matching how real (non-temporary) derivation indices are
  // constructed.
  std::cerr << "step 1: make i1\n";
  Index i1(occ, 1);
  std::cerr << "step 2: make x1\n";
  Index x1(virt, 1);
  std::cerr << "step 3: make a1p\n";
  Index a1p(virt, 2, container::vector<Index>{i1});
  std::cerr << "step 4: a1p built ok, has_proto=" << a1p.has_proto_indices() << "\n";

  std::cerr << "step 5: build C tensor\n";
  auto C = ex<Tensor>(L"C", bra{a1p}, ket{x1});
  std::cerr << "step 6: C built ok\n";

  std::cerr << "step 7: iterate C's indices\n";
  for (const Index &idx : C->as<Tensor>().const_indices()) {
    std::cerr << "  idx label=" << toUtf8(idx.label())
              << " has_proto=" << idx.has_proto_indices() << "\n";
    if (idx.has_proto_indices()) {
      std::cerr << "    proto_indices().size()=" << idx.proto_indices().size()
                << "\n";
      for (const Index &p : idx.proto_indices()) {
        std::cerr << "    proto label=" << toUtf8(p.label()) << "\n";
      }
      std::cerr << "    calling drop_proto_indices()...\n";
      Index dropped = idx.drop_proto_indices();
      std::cerr << "    dropped label=" << toUtf8(dropped.label()) << "\n";
    }
  }
  std::cerr << "step 8: all done, no crash\n";
  return 0;
}
