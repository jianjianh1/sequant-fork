// Isolation experiment (task #19): does a PURE Hadamard-shared-outer (no
// contraction at all) flat x ToT einsum crash too, or only the mixed
// Hadamard+contraction case (g(i,x)*C(i,x;a')->R(i;a'), which segfaults)?
// R3(i;a') = g(i) * C(i;a')  -- i shared+surviving, nothing contracted.
// Calls TA::einsum directly (bypassing the generator) to isolate pure
// TiledArray behavior.
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

  TA::TSpArrayD g = build_sparse_array(world, g_coo, {1}, "g");
  ArrayToT C = build_tot_array<ArrayToT>(world, c_coo, /*outer_rank=*/1,
                                        /*inner_rank=*/1, /*pair_key_rank=*/1,
                                        "C");

  std::cerr << "About to call TA::einsum(g(\"i\"), C(\"i;a\"), \"i;a\")...\n";
  ArrayToT R3;
  R3("i;a") = TA::einsum(g("i"), C("i;a"), "i;a")("i;a");
  world.gop.fence();
  std::cerr << "einsum call completed without crashing.\n";

  std::map<std::pair<int, int>, double> reference;
  {
    std::ifstream in("R3_reference.txt");
    int i, ap;
    double v;
    while (in >> i >> ap >> v) reference[{i, ap}] = v;
  }

  std::map<std::pair<int, int>, double> got;
  for (auto it = R3.begin(); it != R3.end(); ++it) {
    auto ord = it.ordinal();
    if (R3.is_zero(ord) || !R3.is_local(ord)) continue;
    auto fut = R3.find_local(ord);
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
  std::cerr << (ok ? "PASS: pure-Hadamard flat x ToT matches ground truth\n"
                  : "FAIL: mismatch(es)\n");
  return ok ? 0 : 1;
}
