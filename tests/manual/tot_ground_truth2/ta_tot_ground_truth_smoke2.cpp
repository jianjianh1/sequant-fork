// Phase 3, Case 2: numeric ground-truth validation of the OTHER major
// codegen path -- flat x ToT -> ToT Hadamard pass-through (the plain,
// non-DeNest TA::einsum call). Computes R2(i;a') = sum_x g(i,x)*C(i,x;a')
// with the same ragged (2 vs 3 PNOs) synthetic domain as the de-nest case,
// via the actual generated code (generated_case2.h), and diffs every
// (outer,inner) element against an independent numpy reference to machine
// precision.
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>

#include "coo_loader.h"
#include "ta_builder.h"
#include "ta_tensors.h"  // ArrayToT

#include "generated_case2.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

int main(int argc, char** argv) {
  TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  COOTensor g_coo = load_coo("g.coo");
  COOTensor c_coo = load_coo("C.coo");
  if (g_coo.values.empty() || c_coo.values.empty()) {
    std::cerr << "FAIL: could not load g.coo/C.coo (run from tot_smoke2/ "
                 "after gen_data2.py)\n";
    return 1;
  }

  // g(i,x): i must be tiled size-1 to match C's outer tiling convention for
  // the SAME (pair-key) dimension (build_tot_array forces pair-key dims to
  // tile size 1 -- see its own comment) -- a mismatched tiling between two
  // operands sharing an index is what crashed TA::einsum here originally
  // (std::bad_alloc), not a generator bug.
  TA::TSpArrayD g = build_sparse_array(world, g_coo, {1, 4}, "g");
  ArrayToT C = build_tot_array<ArrayToT>(world, c_coo, /*outer_rank=*/2,
                                        /*inner_rank=*/1, /*pair_key_rank=*/1,
                                        "C");

  ArrayToT R2 = compute_R2(g, C);
  world.gop.fence();

  std::map<std::pair<int, int>, double> reference;
  {
    std::ifstream in("R2_reference.txt");
    int i, ap;
    double v;
    while (in >> i >> ap >> v) reference[{i, ap}] = v;
  }
  if (reference.empty()) {
    std::cerr << "FAIL: could not load R2_reference.txt\n";
    return 1;
  }

  // Walk R2's outer tiles/elements, then each inner tensor's own elements
  // (its local a' domain), building up {(i,a') -> value}.
  std::map<std::pair<int, int>, double> got;
  for (auto it = R2.begin(); it != R2.end(); ++it) {
    auto ord = it.ordinal();
    if (R2.is_zero(ord) || !R2.is_local(ord)) continue;
    auto fut = R2.find_local(ord);
    const auto& outer_tile = fut.get();
    for (auto& outer_idx : outer_tile.range()) {
      int i = static_cast<int>(outer_idx[0]);
      const auto& inner = outer_tile[outer_idx];
      for (auto& inner_idx : inner.range()) {
        int ap = static_cast<int>(inner_idx[0]);
        got[{i, ap}] = inner[inner_idx];
      }
    }
  }

  bool ok = true;
  double max_abs_diff = 0.0;
  for (const auto& [key, want] : reference) {
    auto jt = got.find(key);
    if (jt == got.end()) {
      std::cerr << "MISSING (" << key.first << "," << key.second
                << "): expected " << want << "\n";
      ok = false;
      continue;
    }
    double diff = std::abs(jt->second - want);
    max_abs_diff = std::max(max_abs_diff, diff);
    if (diff > 1e-10) {
      std::cerr << "MISMATCH (" << key.first << "," << key.second
                << "): want=" << want << " got=" << jt->second
                << " diff=" << diff << "\n";
      ok = false;
    }
  }
  for (const auto& [key, val] : got) {
    if (!reference.count(key)) {
      std::cerr << "SPURIOUS (" << key.first << "," << key.second
                << "): got=" << val << " (not in reference)\n";
      ok = false;
    }
  }

  std::cerr << "max_abs_diff = " << max_abs_diff << "\n";
  std::cerr << (ok ? "PASS: generated flat->ToT Hadamard pass-through code "
                     "matches numpy ground truth to machine precision\n"
                  : "FAIL: mismatch(es) found\n");
  return ok ? 0 : 1;
}
