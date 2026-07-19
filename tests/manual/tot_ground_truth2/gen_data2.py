#!/usr/bin/env python3
"""Case 2 ground truth: R2(i;a') = sum_x g(i,x) * C(i,x;a') -- the flat x ToT
-> ToT Hadamard pass-through pattern (proto axis a' survives, x contracted).
Reuses the SAME ragged PNO domain (2 for i=0, 3 for i=1) and the same C
values/formula as gen_data.py's de-nest case, plus a new flat g(i,x)."""
import numpy as np

pno_count = {0: 2, 1: 3}
n_x = 4

def gval(i, x):
    return 0.5 + i * 2.0 - x * 0.3

def cval(i, x, ap):
    return 0.1 + i * 1.0 + x * 0.05 + ap * 0.01

with open("g.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            f.write(f"{i} {x} {gval(i, x)}\n")

with open("C.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            for ap in range(pno_count[i]):
                f.write(f"{i} {x} {ap} {cval(i, x, ap)}\n")

with open("R2_reference.txt", "w") as f:
    for i in range(2):
        for ap in range(pno_count[i]):
            s = 0.0
            for x in range(n_x):
                s += gval(i, x) * cval(i, x, ap)
            f.write(f"{i} {ap} {float(s)!r}\n")
            print(f"R2[{i}][{ap}] = {s}")
