#!/usr/bin/env python3
"""Isolation case matching the REAL crash pattern from generated_R1.cpp:
g(i,x) * C(i,x;a') -> R(i;a'): i shared+surviving, x shared+contracted.
(Same as tot_smoke2's original case, just regenerated standalone here.)"""
pno_count = {0: 2, 1: 3}
n_x = 4

def gval(i, x):
    return 0.5 + i * 2.0 - x * 0.3

def cval(i, x, ap):
    return 0.1 + i * 1.0 + x * 0.05 + ap * 0.01

with open("g5.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            f.write(f"{i} {x} {gval(i, x)}\n")

with open("C5.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            for ap in range(pno_count[i]):
                f.write(f"{i} {x} {ap} {cval(i, x, ap)}\n")

with open("R5_reference.txt", "w") as f:
    for i in range(2):
        for ap in range(pno_count[i]):
            s = 0.0
            for x in range(n_x):
                s += gval(i, x) * cval(i, x, ap)
            f.write(f"{i} {ap} {float(s)!r}\n")
