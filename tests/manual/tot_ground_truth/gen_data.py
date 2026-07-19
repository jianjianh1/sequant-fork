#!/usr/bin/env python3
"""Deterministic tiny ragged-PNO test case for Phase 3 ground truth.

T_tot(i; a') : outer i in {0,1}, inner a' ragged per i (2 PNOs for i=0, 3 for i=1)
C_tot(i,x; a'): outer (i,x), i in {0,1} (pair key), x in {0,1,2,3}, inner a' same ragged domain as T
R(i,x) = sum_{a'} T[i,a'] * C[i,x,a']   -- the PNO-to-flat back-transform (de-nest) pattern

Writes T.coo, C.coo (rank-tagged COO text: last column value, rest are indices,
one row per nonzero) and prints the numpy reference R as machine-readable text.
"""
import numpy as np

pno_count = {0: 2, 1: 3}
n_x = 4

def tval(i, ap):
    return 1.0 + i * 10.0 + ap * 3.0

def cval(i, x, ap):
    return 0.1 + i * 1.0 + x * 0.05 + ap * 0.01

with open("T.coo", "w") as f:
    for i in range(2):
        for ap in range(pno_count[i]):
            f.write(f"{i} {ap} {tval(i, ap)}\n")

with open("C.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            for ap in range(pno_count[i]):
                f.write(f"{i} {x} {ap} {cval(i, x, ap)}\n")

R = np.zeros((2, n_x))
for i in range(2):
    for x in range(n_x):
        s = 0.0
        for ap in range(pno_count[i]):
            s += tval(i, ap) * cval(i, x, ap)
        R[i, x] = s

with open("R_reference.txt", "w") as f:
    for i in range(2):
        for x in range(n_x):
            f.write(f"{i} {x} {float(R[i,x])!r}\n")

print("R reference:")
print(R)
