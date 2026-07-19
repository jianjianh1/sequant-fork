// Test whether the DIRECT expression syntax (C(ann) = A * B, matching
// TiledArray's own tests/general_product.cpp::expression_general_product_*
// tests, e.g. the literally-named "csv_like" one) avoids the crash that
// TA::einsum(...) hits for this same shape (task #19).
// R4(i;a') = sum_x g(x) * C(i,x;a') -- same case as
// ta_isolate_contraction_only.cpp (confirmed segfault via TA::einsum).
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

  std::cerr << "About to call R4(\"i;a\") = g(\"x\") * C(\"i,x;a\") (direct "
               "expression, no TA::einsum)...\n";
  ArrayToT R4;
  R4("i;a") = g("x") * C("i,x;a");
  world.gop.fence();
  std::cerr << "direct expression completed without crashing.\n";

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
  std::cerr << (ok ? "PASS: direct expression matches ground truth\n"
                  : "FAIL: mismatch(es)\n");
  return ok ? 0 : 1;
}
