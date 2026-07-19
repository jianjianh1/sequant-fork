// Test whether adding an external (one-sided, only-in-flat-operand) index
// avoids the crash: R(i,y;a') = sum_x g(i,x,y) * C(i,x;a'). h={i}
// (nonempty), e={y} (external, only in g), contracted={x}.
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

  COOTensor g_coo = load_coo("g6.coo");
  COOTensor c_coo = load_coo("C6.coo");
  if (g_coo.values.empty() || c_coo.values.empty()) {
    std::cerr << "FAIL: could not load g6.coo/C6.coo\n";
    return 1;
  }

  TA::TSpArrayD g = build_sparse_array(world, g_coo, {1, 4, 2}, "g");
  ArrayToT C = build_tot_array<ArrayToT>(world, c_coo, /*outer_rank=*/2,
                                        /*inner_rank=*/1, /*pair_key_rank=*/1,
                                        "C");

  std::cerr << "About to call einsum(g(\"i,x,y\"), C(\"i,x;a\"), "
               "\"i,y;a\")...\n";
  ArrayToT R = TA::einsum(g("i,x,y"), C("i,x;a"), "i,y;a");
  world.gop.fence();
  std::cerr << "einsum completed without crashing.\n";

  std::map<std::tuple<int, int, int>, double> reference;
  {
    std::ifstream in("R6_reference.txt");
    int i, y, ap;
    double v;
    while (in >> i >> y >> ap >> v) reference[{i, y, ap}] = v;
  }

  std::map<std::tuple<int, int, int>, double> got;
  for (auto it = R.begin(); it != R.end(); ++it) {
    auto ord = it.ordinal();
    if (R.is_zero(ord) || !R.is_local(ord)) continue;
    auto fut = R.find_local(ord);
    const auto& outer_tile = fut.get();
    for (auto& outer_idx : outer_tile.range()) {
      int i = static_cast<int>(outer_idx[0]);
      int y = static_cast<int>(outer_idx[1]);
      const auto& inner = outer_tile[outer_idx];
      for (auto& inner_idx : inner.range()) {
        int ap = static_cast<int>(inner_idx[0]);
        got[{i, y, ap}] = inner[inner_idx];
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
  std::cerr << (ok ? "PASS: external-variant matches ground truth\n"
                  : "FAIL: mismatch(es)\n");
  return ok ? 0 : 1;
}
