// Isolation experiment (task #19), second half: PURE contraction, no
// Hadamard-shared outer index at all. R4(i;a') = sum_x g(x) * C(i,x;a') --
// x is shared+contracted between g and C; i is NOT shared with g (only
// appears in C, passes straight through to the result). Companion to
// ta_isolate_hadamard_only.cpp (which confirmed pure-Hadamard, no
// contraction, does NOT crash).
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

  COOTensor g_coo = load_coo("g.coo");
  COOTensor c_coo = load_coo("C.coo");
  if (g_coo.values.empty() || c_coo.values.empty()) {
    std::cerr << "FAIL: could not load g.coo/C.coo\n";
    return 1;
  }

  TA::TSpArrayD g = build_sparse_array(world, g_coo, {4}, "g");
  ArrayToT C = build_tot_array<ArrayToT>(world, c_coo, /*outer_rank=*/2,
                                        /*inner_rank=*/1, /*pair_key_rank=*/1,
                                        "C");

  std::cerr << "About to call TA::einsum(g(\"x\"), C(\"i,x;a\"), \"i;a\")...\n";
  ArrayToT R4;
  R4("i;a") = TA::einsum(C("i,x;a"), g("x"), "i;a")("i;a");
  world.gop.fence();
  std::cerr << "einsum call completed without crashing.\n";

  std::map<std::pair<int, int>, double> reference;
  {
    std::ifstream in("R4_reference.txt");
    int i, ap;
    double v;
    while (in >> i >> ap >> v) reference[{i, ap}] = v;
  }

  std::map<std::pair<int, int>, double> got;
  for (auto it = R4.begin(); it != R4.end(); ++it) {
    auto ord = it.ordinal();
    if (R4.is_zero(ord) || !R4.is_local(ord)) continue;
    auto fut = R4.find_local(ord);
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
  for (const auto& [key, want] : reference) {
    auto jt = got.find(key);
    if (jt == got.end()) {
      std::cerr << "MISSING (" << key.first << "," << key.second << ")\n";
      ok = false;
      continue;
    }
    double diff = std::abs(jt->second - want);
    if (diff > 1e-10) {
      std::cerr << "MISMATCH (" << key.first << "," << key.second
                << "): want=" << want << " got=" << jt->second << "\n";
      ok = false;
    }
  }
  std::cerr << (ok ? "PASS: pure-contraction flat x ToT matches ground truth\n"
                  : "FAIL: mismatch(es)\n");
  return ok ? 0 : 1;
}
