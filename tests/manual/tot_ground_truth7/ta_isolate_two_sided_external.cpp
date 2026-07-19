// Test the TWO-SIDED external pattern matching TiledArray's own PASSING
// expression_general_product_csv_like test structure exactly:
//   R(i,y,z;a') = sum_x g(i,x,y) * C(i,x,z;a')
// h={i} nonempty, e={y,z} (y external-only-to-g, z external-only-to-C),
// contracted={x}. If this is the missing ingredient, this should NOT crash.
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>

#include "coo_loader.h"
#include "ta_builder.h"
#include "ta_tensors.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

int main(int argc, char** argv) {
  TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  COOTensor g_coo = load_coo("g7.coo");
  COOTensor c_coo = load_coo("C7.coo");
  if (g_coo.values.empty() || c_coo.values.empty()) {
    std::cerr << "FAIL: could not load g7.coo/C7.coo\n";
    return 1;
  }

  TA::TSpArrayD g = build_sparse_array(world, g_coo, {1, 3, 2}, "g");
  ArrayToT C = build_tot_array<ArrayToT>(world, c_coo, /*outer_rank=*/3,
                                        /*inner_rank=*/1, /*pair_key_rank=*/1,
                                        "C");

  std::cerr << "About to call einsum(g(\"i,x,y\"), C(\"i,x,z;a\"), "
               "\"i,y,z;a\")...\n";
  ArrayToT R = TA::einsum(g("i,x,y"), C("i,x,z;a"), "i,y,z;a");
  world.gop.fence();
  std::cerr << "einsum completed without crashing.\n";

  std::map<std::tuple<int, int, int, int>, double> reference;
  {
    std::ifstream in("R7_reference.txt");
    int i, y, z, ap;
    double v;
    while (in >> i >> y >> z >> ap >> v) reference[{i, y, z, ap}] = v;
  }

  std::map<std::tuple<int, int, int, int>, double> got;
  for (auto it = R.begin(); it != R.end(); ++it) {
    auto ord = it.ordinal();
    if (R.is_zero(ord) || !R.is_local(ord)) continue;
    auto fut = R.find_local(ord);
    const auto& outer_tile = fut.get();
    for (auto& outer_idx : outer_tile.range()) {
      int i = static_cast<int>(outer_idx[0]);
      int y = static_cast<int>(outer_idx[1]);
      int z = static_cast<int>(outer_idx[2]);
      const auto& inner = outer_tile[outer_idx];
      for (auto& inner_idx : inner.range()) {
        int ap = static_cast<int>(inner_idx[0]);
        got[{i, y, z, ap}] = inner[inner_idx];
      }
    }
  }

  bool ok = true;
  for (const auto& [key, want] : reference) {
    auto jt = got.find(key);
    if (jt == got.end()) {
      std::cerr << "MISSING\n";
      ok = false;
      continue;
    }
    double diff = std::abs(jt->second - want);
    if (diff > 1e-10) {
      std::cerr << "MISMATCH: want=" << want << " got=" << jt->second << "\n";
      ok = false;
    }
  }
  std::cerr << (ok ? "PASS: two-sided-external matches ground truth\n"
                  : "FAIL: mismatch(es)\n");
  return ok ? 0 : 1;
}
