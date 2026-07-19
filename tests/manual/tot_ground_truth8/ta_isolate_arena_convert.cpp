// Decisive experiment (task #19): convert the plain-Tensor ToT operand C
// into an Arena-backed ToT array (matching TA's own passing csv_like
// test's array type) via TA::foreach + arena_trivial_unary, THEN run the
// exact same einsum call that crashes with the plain-Tensor C. If this
// avoids the crash and matches ground truth, it validates the fix path:
// convert real C/t ToT arrays to Arena inner cells before this contraction
// shape, then convert the result back.
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <TiledArray/tensor/arena_tensor.h>
#include <TiledArray/tensor/arena_kernels.h>

#include "coo_loader.h"
#include "ta_builder.h"
#include "ta_tensors.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

using ArenaInner = TA::ArenaTensor<double, TA::Range>;
using ArenaOuter = TA::Tensor<ArenaInner>;
using ArenaSpArr = TA::DistArray<ArenaOuter, TA::SparsePolicy>;

// Converts one plain outer tile (Tensor<Tensor<double>>) into an Arena
// outer tile (Tensor<ArenaTensor<double,Range>>), copying every inner cell.
ArenaOuter to_arena_tile(const TA::Tensor<TA::Tensor<double>>& src) {
  auto fill = [](double* dst, const double* src_ptr, std::size_t n) {
    std::copy(src_ptr, src_ptr + n, dst);
  };
  return TA::detail::arena_trivial_unary<ArenaOuter>(src, fill);
}

int main(int argc, char** argv) {
  TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  COOTensor g_coo = load_coo("g5.coo");
  COOTensor c_coo = load_coo("C5.coo");
  if (g_coo.values.empty() || c_coo.values.empty()) {
    std::cerr << "FAIL: could not load g5.coo/C5.coo\n";
    return 1;
  }

  TA::TSpArrayD g = build_sparse_array(world, g_coo, {1, 4}, "g");
  ArrayToT C_plain = build_tot_array<ArrayToT>(
      world, c_coo, /*outer_rank=*/2, /*inner_rank=*/1, /*pair_key_rank=*/1,
      "C");

  std::cerr << "Converting C to Arena-backed form...\n";
  ArenaSpArr C_arena = TA::foreach<ArenaOuter>(
      C_plain, [](ArenaOuter& out_tile, const TA::Tensor<TA::Tensor<double>>& in_tile) {
        out_tile = to_arena_tile(in_tile);
      });
  world.gop.fence();
  std::cerr << "Conversion done. About to call einsum(g(\"i,x\"), "
               "C_arena(\"i,x;a\"), \"i;a\")...\n";

  ArenaSpArr R = TA::einsum(g("i,x"), C_arena("i,x;a"), "i;a");
  world.gop.fence();
  std::cerr << "einsum with Arena-backed C completed without crashing.\n";

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
      for (std::size_t e = 0; e < inner.range().volume(); ++e) {
        auto iix = inner.range().idx(e);
        int ap = static_cast<int>(iix[0]);
        got[{i, ap}] = inner.data()[e];
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
  std::cerr << (ok ? "PASS: Arena-converted C matches ground truth -- FIX "
                     "CONFIRMED\n"
                  : "FAIL: mismatch(es)\n");
  return ok ? 0 : 1;
}
