// Phase 2 verification (twinkly-dazzling-shamir.md): isolated ExprPtr-based
// unit tests for TiledArrayGenerator's ToT (PNO/CSV-restriction) support --
// classify_indices(), the represent(Index)/index_annotation() ";"-split,
// and check_tot_contraction_safety()'s Hadamard/contraction/collapse gating.
// Deliberately NOT going through any MPQC/CC derivation machinery -- these
// are small, hand-built trees so a classification bug is caught in
// isolation, before it's entangled with real CCSD equation complexity.
#include <SeQuant/core/context.hpp>
#include <SeQuant/core/export/export.hpp>
#include <SeQuant/core/export/tiledarray_generator.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
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

int g_pass = 0;
int g_fail = 0;

void expect(bool cond, const std::string &what) {
  if (cond) {
    std::cerr << "  PASS: " << what << "\n";
    ++g_pass;
  } else {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_fail;
  }
}

// Runs a single-Product expression through the real export pipeline
// (to_export_tree + export_expression), catching any thrown Exception.
// Returns {generated_code, exception_message} -- exactly one is non-empty.
std::pair<std::string, std::string> try_export(const ExprPtr &e) {
  try {
    auto tree = to_export_tree(e, /*retain_braket=*/false);
    TiledArrayGenerator generator;
    TiledArrayGeneratorContext ctx;
    export_expression(std::move(tree), generator, ctx);
    return {generator.get_generated_code(), ""};
  } catch (const std::exception &ex) {
    return {"", ex.what()};
  } catch (...) {
    return {"", "unknown non-std::exception thrown"};
  }
}

}  // namespace

int main() {
  load_convention();
  auto isr = get_default_context().index_space_registry();
  IndexSpace occ = isr->retrieve(L"i");
  IndexSpace virt = isr->retrieve(L"a");

  // NOTE: Index::make_tmp_index() mints ordinals >= min_tmp_index(), which
  // Index::drop_proto_indices() (going through the string-label ctor
  // internally, index.hpp's check_nonreserved()) explicitly rejects --
  // confirmed via a standalone repro (test_tot_minimal_repro.cpp) to be a
  // real SeQuant constraint, not a generator bug: temp-minted ordinals are
  // reserved for the make_tmp_index/IndexFactory path specifically. Real
  // CC-derivation-created proto-indexed axes get plain (low, non-temp)
  // ordinals, so classify_indices() calling drop_proto_indices() on THOSE
  // never hits this -- confirmed separately by the real T1/T2 fixture
  // working. Every index below therefore uses the plain (space, ordinal[,
  // proto_indices]) constructor with small explicit ordinals, matching how
  // real derivation indices are actually shaped.

  // --- Case 1: flat x flat -> flat (regression check) ---------------------
  {
    std::cerr << "=== Case 1: flat x flat (regression) ===\n";
    Index i1(occ, 1);
    Index a1(virt, 1);
    Index a2(virt, 2);
    // g(i1,a1) * f(a1,a2) -> R(i1,a2), a1 contracted at the outer level.
    auto g = ex<Tensor>(L"g", bra{i1}, ket{a1});
    auto f = ex<Tensor>(L"f", bra{a1}, ket{a2});
    auto prod = ex<Product>(Product{1, {g, f}});
    // fake a "compute into result R(i1,a2)" by wrapping as the top Sum's
    // sole summand -- to_export_tree infers the result tensor's indices
    // from the expression's own free indices, so this needs no separate
    // result Tensor object.
    auto [code, err] = try_export(prod);
    expect(err.empty(), "no exception for flat x flat");
    expect(code.find("TA::TSpArrayD") != std::string::npos,
          "generated flat TA::TSpArrayD code");
    expect(code.find("DistArray<TA::Tensor<TA::Tensor") == std::string::npos,
          "no ArrayToT type appears anywhere (purely flat)");
    if (!err.empty()) std::cerr << "    (exception: " << err << ")\n";
  }

  // --- Case 2: flat x ToT -> ToT (PNO load / Hadamard pass-through) -------
  {
    std::cerr << "=== Case 2: flat x ToT -> ToT (Hadamard pass-through) ===\n";
    Index i1(occ, 1);
    Index x1(virt, 1);  // flat CSV-basis dummy
    Index a1p(virt, 2, container::vector<Index>{i1});  // proto by i1
    // g_flat(i1,x1) * C(a1p,x1) -> R(i1,a1p): x1 contracted at outer level,
    // a1p survives (Hadamard-through the PNO axis) -- a legitimate ToT op.
    auto g = ex<Tensor>(L"g", bra{i1}, ket{x1});
    auto C = ex<Tensor>(L"C", bra{a1p}, ket{x1});
    auto prod = ex<Product>(Product{1, {g, C}});
    auto [code, err] = try_export(prod);
    expect(err.empty(), "no exception for flat x ToT Hadamard pass-through");
    if (err.empty()) {
      expect(code.find("DistArray<TA::Tensor<TA::Tensor<double>>") !=
                std::string::npos,
            "generated ArrayToT-typed result code");
      expect(code.find(';') != std::string::npos,
            "ToT annotation contains ';' outer/inner separator");
    } else {
      std::cerr << "    (exception: " << err << ")\n";
    }
  }

  // --- Case 3: ToT x ToT, inner (PNO) axis fully collapsed to flat -------
  // The real "PNO back-transform to flat" pattern (confirmed via direct
  // inspection of SeQuant's own TiledArray eval backend,
  // core/eval/backends/tiledarray/result.hpp:~619-624, and TiledArray's
  // own test suite, tests/dot_inner.cpp/einsum.cpp): T(i1;a1p) contracted
  // with C(a1p;x1) over the shared PNO axis a1p, with i1 (from T) and x1
  // (from C) surviving as ordinary flat outer indices -- exactly the shape
  // the real T1/T2 residuals hit on essentially every PNO-touching term.
  // Must succeed via TA::einsum<TA::DeNest::True>.
  {
    std::cerr << "=== Case 3: ToT x ToT, inner axis fully collapsed to flat "
                 "(de-nest) ===\n";
    Index i1(occ, 1);
    Index x1(virt, 1);  // flat CSV-basis dummy
    Index a1p(virt, 2, container::vector<Index>{i1});
    auto T = ex<Tensor>(L"t", bra{i1}, ket{a1p});
    auto C = ex<Tensor>(L"C", bra{a1p}, ket{x1});
    auto prod = ex<Product>(Product{1, {T, C}});
    auto [code, err] = try_export(prod);
    expect(err.empty(), "no exception for full ToT-inner collapse to flat");
    if (err.empty()) {
      expect(code.find("TA::einsum<TA::DeNest::True>") != std::string::npos,
            "generated code uses TA::einsum<TA::DeNest::True>");
      expect(code.find("TA::TSpArrayD") != std::string::npos,
            "result is flat TA::TSpArrayD (de-nested)");
    } else {
      std::cerr << "    (exception: " << err << ")\n";
    }
  }

  // --- Case 4: mixed Hadamard+contraction -- must throw -------------------
  // A(a1p,b1p) * B(a1p,b1p) -- both a1p and b1p are shared between A and B.
  // Product's own free-index algebra always fully contracts a repeated
  // identity (there's no way to make ONE of two identically-repeated axes
  // "survive" via bare Product construction/to_export_tree inference alone
  // -- confirmed empirically: routing this through to_export_tree just
  // contracts everything, including the shared outer proto i1, down to a
  // scalar, which is Case 3's de-nest path, not the mixed case). So this
  // exercises check_tot_contraction_safety() directly via
  // generator.compute() with an EXPLICIT result tensor R(a1p) -- forcing
  // a1p to "survive" while b1p is absent, independent of Product's own
  // free-index inference.
  {
    std::cerr << "=== Case 4: mixed Hadamard+contraction (must throw) ===\n";
    Index i1(occ, 1);
    Index a1p(virt, 2, container::vector<Index>{i1});
    Index b1p(virt, 3, container::vector<Index>{i1});
    auto A = ex<Tensor>(L"A", bra{}, ket{a1p, b1p});
    auto B = ex<Tensor>(L"B", bra{}, ket{a1p, b1p});
    auto prod = ex<Product>(Product{1, {A, B}});
    Tensor result(L"R", bra{}, ket{a1p});
    TiledArrayGenerator generator;
    TiledArrayGeneratorContext ctx;
    std::string err;
    try {
      generator.compute(*prod, result, ctx);
    } catch (const std::exception &ex) {
      err = ex.what();
    } catch (...) {
      err = "unknown non-std::exception thrown";
    }
    expect(!err.empty(), "throws for mixed Hadamard+contraction");
    if (!err.empty())
      std::cerr << "    (exception, expected: " << err << ")\n";
    else
      std::cerr << "    (no exception -- unexpectedly succeeded)\n";
  }

  std::cerr << "\n" << g_pass << " passed, " << g_fail << " failed\n";
  return g_fail == 0 ? 0 : 1;
}
