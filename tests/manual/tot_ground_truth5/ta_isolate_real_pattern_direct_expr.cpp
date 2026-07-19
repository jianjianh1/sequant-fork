// Test the DIRECT expression syntax for the pattern that ACTUALLY matches
// the real generated_R1.cpp crash line:
//   I_ap1("i;a") = TA::einsum(I_i_x("i,x"), C_ap1_x("i,x;a"), "i;a")
// h={i} (nonempty, shared+surviving), e=empty, contracted={x}. This is a
// DIFFERENT branch inside TA::einsum than tot_smoke4's h=empty case (falls
// to the manual "hadamard reduction"/"generalized contraction" code, not
// the `if (!h)` A*B delegation) -- so it needs its own direct test.
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

  COOTensor g_coo = load_coo("g5.coo");
  COOTensor c_coo = load_coo("C5.coo");
  if (g_coo.values.empty() || c_coo.values.empty()) {
    std::cerr << "FAIL: could not load g5.coo/C5.coo\n";
    return 1;
  }

  // NOTE: must match build_tot_array's own adaptive_tile_sizes() choice for
  // the "x" dimension (extent 4, target_tiles_per_dim=8 -> tile size 1, i.e.
  // 4 separate tiles), not an arbitrarily-chosen tile size -- a genuine
  // TiledRange1 mismatch between g and C on the shared/contracted dimension
  // is what TA::einsum's debug-mode congruence check (cont_engine.h) caught
  // (see below); a Release build has this check compiled out (#ifndef
  // NDEBUG) and silently proceeds instead, which is the real reason this
  // looked like an internal TA crash rather than an ordinary setup bug.
  TA::TSpArrayD g = build_sparse_array(world, g_coo, {1, 1}, "g");
  ArrayToT C = build_tot_array<ArrayToT>(world, c_coo, /*outer_rank=*/2,
                                        /*inner_rank=*/1, /*pair_key_rank=*/1,
                                        "C");

  std::cerr << "About to call R(\"i;a\") = g(\"i,x\") * C(\"i,x;a\") (direct "
               "expression)...\n";
  ArrayToT R;
  R("i;a") = g("i,x") * C("i,x;a");
  world.gop.fence();
  std::cerr << "direct expression completed without crashing.\n";

  std::map<std::pair<int, int>, double> reference;
  {
    std::ifstream in("R5_reference.txt");
    int i, ap;
    double v;
    while (in >> i >> ap >> v) reference[{i, ap}] = v;
  }

  std::map<std::pair<int, int>, double> got;
  for (auto it = R.begin(); it != R.end(); ++it) {
    auto ord = it.ordinal();
    if (R.is_zero(ord) || !R.is_local(ord)) continue;
    auto fut = R.find_local(ord);
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
  std::cerr << (ok ? "PASS: real-pattern direct expression matches ground truth\n"
                  : "FAIL: mismatch(es)\n");
  return ok ? 0 : 1;
}
