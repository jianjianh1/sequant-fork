// Minimal, standalone, COMPILED repro for a SECOND SeQuant core bug, found
// while investigating a real generated-code crash on top of the ddd208d5 fix
// ("eval: preserve domain-tag indices with 3+ occurrences during
// binarization").
//
// ddd208d5 fixed eval_expr.cpp's make_prod so that an index occurring MORE
// than twice in a term (a "domain tag", e.g. the CSV/PNO occupied-pair index
// i,j that legitimately never gets Wick-contracted) is forced to survive a
// binarization node's target-index computation, even when the node's own
// local two-branch view (uncontracted_idxs, from
// utility/indices.hpp's left_to_right_binarization_indices) would otherwise
// drop it. That fix's trigger is a plain absolute threshold: "this node's
// cumulative subfacs occurrence count for index k exceeds 2" -- computed once
// via get_used_indices_with_counts(ex<Product>(subfacs)), where subfacs is
// the full set of ORIGINAL leaf tensors already folded into this node's
// subtree (collect_tensor_factors() walks all the way down through any
// previously-built Product-type descendant).
//
// That threshold is necessary but not sufficient: it has no way to tell
// "count > 2 because more occurrences of k still live in an unmerged sibling
// subtree elsewhere in the tree" (must keep forcing survival) apart from
// "count > 2 because ALL of k's whole-term occurrences already happen to be
// gathered right here" (nothing left to combine with -- k should now behave
// like any other exhausted dummy: contract away unless separately flagged
// external). The current code can only ever answer "yes, forced" once the
// count first exceeds 2, and that answer never reverts, because subfacs only
// grows monotonically as more nodes are combined going up the tree. So an
// index that is genuinely, validly summed over 3+ tensor factors (NOT an
// external/domain-tag index -- it must vanish from the term's own final
// result) gets stuck "surviving" forever past the point where it should have
// been contracted, and leaks into a LATER combination with a tensor that
// doesn't carry it at all and a target that doesn't want it either --
// exactly the runtime shape hit in ta-bench/src/generated_t1_residual.cpp:
// `TA::einsum(A(...,k,...), B(...no k...), "...no k...")(...)` -- an index
// present in exactly one operand's OWN annotation, absent from the sibling
// operand, and absent from the target: not a valid bilinear contraction.
//
// Repro shape: k occurs in exactly 3 leaf factors (A, B, C) -- structurally
// indistinguishable, at binarization time, from the domain-tag case ddd208d5
// targets. But here k has NO other occurrences anywhere: once A, B, and C
// have all been combined, EVERY occurrence of k has been gathered, and nothing
// else in the term ever needs it again. The result should not carry k. A 4th
// factor E shares k's "outer" partners p,q,r (not k itself) and contributes
// one genuinely free index s. Hand-parenthesized as strictly left-associative
// ((A*B)*C)*E (matching the shape actually seen in the generated CCSD code) so
// the bug is deterministic and doesn't depend on what optimize()'s DP would
// pick.
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

  // k: occurs in A, B, C -- exactly 3 times, nowhere else. A genuinely
  // fully-summed dummy (never an external/domain-tag index), just one that
  // happens to touch 3 factors instead of the textbook 2.
  Index k(occ, 1);
  // p, q, r: each occurs exactly twice (once in A/B/C, once in E) -- normal
  // 2-occurrence Wick dummies.
  Index p(virt, 1);
  Index q(virt, 2);
  Index r(virt, 3);
  // s: the term's one genuine external index (occurs once).
  Index s(virt, 4);

  auto A = ex<Tensor>(L"A", bra{k}, ket{p});
  auto B = ex<Tensor>(L"B", bra{k}, ket{q});
  auto C = ex<Tensor>(L"C", bra{k}, ket{r});
  auto E = ex<Tensor>(L"E", bra{p, q, r}, ket{s});

  std::cout << "Factors: A(k;p)=" << toUtf8(A->to_latex())
            << "  B(k;q)=" << toUtf8(B->to_latex())
            << "  C(k;r)=" << toUtf8(C->to_latex())
            << "  E(p,q,r;s)=" << toUtf8(E->to_latex()) << "\n";
  std::cout << "k's true whole-term occurrence count is 3 (A, B, C only) -- "
               "identical in shape to a domain tag at binarization time, but "
               "k must NOT appear in the final result (only s should).\n";

  // Hand-parenthesize as strictly left-associative ((A*B)*C)*E -- the exact
  // shape observed in the real generated code (an intermediate accumulates
  // all 3 occurrences of the "domain-tag-shaped" index, then gets combined
  // with one more factor that doesn't have it).
  auto AB = ex<Product>(Product{1, {A, B}, Product::Flatten::No});
  auto ABC = ex<Product>(Product{1, {AB, C}, Product::Flatten::No});
  auto whole = ex<Product>(Product{1, {ABC, E}, Product::Flatten::No});

  std::cout << "\nHand-parenthesized expression: " << toUtf8(whole->to_latex())
            << "\n";

  // The enclosing equation's TRUE external/free index set is {s} ONLY --
  // k, p, q, r are all genuine (term-local) Wick-contraction dummies, even
  // though k happens to recur 3 times within this one term for unrelated
  // combinatorial reasons (see file header). Supplying this explicitly is
  // what lets make_prod's fix (eval_expr.cpp) tell this case apart from the
  // structurally-identical domain-tag case in
  // test_domain_tag_early_gather.cpp, where the recurring index instead
  // belongs to this set and must survive.
  IndexSet const external{s};

  std::string code = export_and_print(
      whole, "((A*B)*C)*E, hand-parenthesized", external);

  bool has_k = code.find("k_1") != std::string::npos;
  bool has_s = code.find("a_4") != std::string::npos;  // s's rendered label
  std::cout << "\n=== VERDICT ===\n";
  std::cout << "k appears anywhere in generated code: " << has_k << "  "
               "(expected: false -- k must be fully contracted)\n";
  std::cout << "s (a_4) appears anywhere in generated code: " << has_s
            << "  (expected: true -- s is the term's only real external)\n";
  if (code.empty()) {
    std::cout << "Export THREW rather than producing code -- this is the bug "
                 "manifesting as a hard failure (mirrors the real "
                 "TA::einsum \"not a permutation\" exception): an orphaned "
                 "index present in one operand, absent from its sibling, "
                 "absent from the target.\n";
  } else if (has_k) {
    std::cout << "BUG REPRODUCED: k leaked into the generated code even "
                 "though every one of its 3 occurrences (A, B, C) was long "
                 "since combined and no further use of k exists anywhere in "
                 "the term.\n";
  } else {
    std::cout << "k correctly vanished -- no bug observed with this build.\n";
  }

  return 0;
}
