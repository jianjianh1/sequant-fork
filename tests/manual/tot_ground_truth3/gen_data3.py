#!/usr/bin/env python3
"""Isolation case A: PURE Hadamard shared-outer, no contraction at all.
R3(i;a') = g(i) * C(i;a')  -- i shared+surviving, NOTHING contracted.
Ragged PNO domain reused (2 for i=0, 3 for i=1)."""
pno_count = {0: 2, 1: 3}

def gval(i):
    return 2.0 + i * 5.0

def cval(i, ap):
    return 0.1 + i * 1.0 + ap * 0.01

with open("g.coo", "w") as f:
    for i in range(2):
        f.write(f"{i} {gval(i)}\n")

with open("C.coo", "w") as f:
    for i in range(2):
        for ap in range(pno_count[i]):
            f.write(f"{i} {ap} {cval(i, ap)}\n")

with open("R3_reference.txt", "w") as f:
    for i in range(2):
        for ap in range(pno_count[i]):
            v = gval(i) * cval(i, ap)
            f.write(f"{i} {ap} {float(v)!r}\n")
            print(f"R3[{i}][{ap}] = {v}")
