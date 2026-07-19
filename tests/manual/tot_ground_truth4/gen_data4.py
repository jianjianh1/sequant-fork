#!/usr/bin/env python3
"""Isolation case B: PURE contraction, no Hadamard-shared outer index.
R4(i;a') = sum_x g(x) * C(i,x;a') -- x shared+contracted; i is NOT shared
with g at all (g has no i dependence), just passes through from C to R."""
pno_count = {0: 2, 1: 3}
n_x = 4

def gval(x):
    return 1.5 - x * 0.2

def cval(i, x, ap):
    return 0.1 + i * 1.0 + x * 0.05 + ap * 0.01

with open("g.coo", "w") as f:
    for x in range(n_x):
        f.write(f"{x} {gval(x)}\n")

with open("C.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            for ap in range(pno_count[i]):
                f.write(f"{i} {x} {ap} {cval(i, x, ap)}\n")

with open("R4_reference.txt", "w") as f:
    for i in range(2):
        for ap in range(pno_count[i]):
            s = 0.0
            for x in range(n_x):
                s += gval(x) * cval(i, x, ap)
            f.write(f"{i} {ap} {float(s)!r}\n")
            print(f"R4[{i}][{ap}] = {s}")
