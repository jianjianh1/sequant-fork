// Phase 3 (twinkly-dazzling-shamir.md): numeric ground-truth validation of
// the native SeQuant TiledArrayGenerator's ToT codegen. Computes
//   R(i,x) = sum_a' T(i;a') * C(i,x;a')
// (the PNO-to-flat back-transform / de-nest pattern, the single riskiest
// unverified call this generator emits) using a tiny, deterministic, RAGGED
// (per-i PNO count 2 vs 3) synthetic dataset, via the ACTUAL generated
// C++ (generated_case.h, produced by gen_tot_smoke_case.cpp from the exact
// same T(i;a')*C(a';x) expression used in Phase 2's isolated Case 3 unit
// test) run against real TiledArray, and diffs the result to machine
// precision against gen_data.py's independent numpy reference
// (R_reference.txt) -- deliberately computed without TiledArray, to avoid
// a shared-bug blind spot.
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>

#include "coo_loader.h"
#include "ta_builder.h"
#include "ta_tensors.h"  // ArrayToT

#include "generated_case.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

int main(int argc, char** argv) {
  TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  COOTensor t_coo = load_coo("T.coo");
  COOTensor c_coo = load_coo("C.coo");
  if (t_coo.values.empty() || c_coo.values.empty()) {
    std::cerr << "FAIL: could not load T.coo/C.coo (run from tot_smoke/ "
                 "after gen_data.py)\n";
    return 1;
  }

  ArrayToT T = build_tot_array<ArrayToT>(world, t_coo, /*outer_rank=*/1,
                                        /*inner_rank=*/1, /*pair_key_rank=*/1,
                                        "T");
  ArrayToT C = build_tot_array<ArrayToT>(world, c_coo, /*outer_rank=*/2,
                                        /*inner_rank=*/1, /*pair_key_rank=*/1,
                                        "C");

  TA::TSpArrayD R = compute_R(T, C);
  world.gop.fence();

  // Load the independent numpy reference: lines "i x value".
  std::map<std::pair<int, int>, double> reference;
  {
    std::ifstream in("R_reference.txt");
    int i, x;
    double v;
    while (in >> i >> x >> v) reference[{i, x}] = v;
  }
  if (reference.empty()) {
    std::cerr << "FAIL: could not load R_reference.txt\n";
    return 1;
  }

  // Pull R's values out via TA::Tensor conversion (small array -- just
  // gather every tile to rank 0 for a direct element-by-element diff).
  std::map<std::pair<int, int>, double> got;
  for (auto it = R.begin(); it != R.end(); ++it) {
    if (R.is_zero(it.ordinal())) continue;
    TA::Tensor<double> tile = R.find(it.ordinal()).get();
    const auto& rng = tile.range();
    for (auto& idx : rng) {
      got[{static_cast<int>(idx[0]), static_cast<int>(idx[1])}] = tile[idx];
    }
  }

  bool ok = true;
  double max_abs_diff = 0.0;
  for (const auto& [key, want] : reference) {
    auto it = got.find(key);
    if (it == got.end()) {
      std::cerr << "MISSING (" << key.first << "," << key.second
                << "): expected " << want << "\n";
      ok = false;
      continue;
    }
    double diff = std::abs(it->second - want);
    max_abs_diff = std::max(max_abs_diff, diff);
    if (diff > 1e-10) {
      std::cerr << "MISMATCH (" << key.first << "," << key.second
                << "): want=" << want << " got=" << it->second
                << " diff=" << diff << "\n";
      ok = false;
    }
  }
  // Also flag any GOT entry the reference didn't expect (would mean the
  // generated code produced spurious nonzero elements).
  for (const auto& [key, val] : got) {
    if (!reference.count(key)) {
      std::cerr << "SPURIOUS (" << key.first << "," << key.second
                << "): got=" << val << " (not in reference)\n";
      ok = false;
    }
  }

  std::cerr << "max_abs_diff = " << max_abs_diff << "\n";
  std::cerr << (ok ? "PASS: generated ToT->flat de-nest code matches numpy "
                     "ground truth to machine precision\n"
                  : "FAIL: mismatch(es) found\n");
  return ok ? 0 : 1;
}
