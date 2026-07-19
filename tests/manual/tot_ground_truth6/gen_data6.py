#!/usr/bin/env python3
"""Two-sided-external variant, mirroring TA's own passing csv_like test
structure: g(i,x,y) * C(i,x;a') -> R(i,y;a'). h={i} nonempty, e={y}
(external, only in g -- one-sided, not two-sided like csv_like, but let's
see if ANY external index at all changes the outcome), contracted={x}."""
pno_count = {0: 2, 1: 3}
n_x = 4
n_y = 2

def gval(i, x, y):
    return 0.5 + i * 2.0 - x * 0.3 + y * 0.7

def cval(i, x, ap):
    return 0.1 + i * 1.0 + x * 0.05 + ap * 0.01

with open("g6.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            for y in range(n_y):
                f.write(f"{i} {x} {y} {gval(i, x, y)}\n")

with open("C6.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            for ap in range(pno_count[i]):
                f.write(f"{i} {x} {ap} {cval(i, x, ap)}\n")

with open("R6_reference.txt", "w") as f:
    for i in range(2):
        for y in range(n_y):
            for ap in range(pno_count[i]):
                s = 0.0
                for x in range(n_x):
                    s += gval(i, x, y) * cval(i, x, ap)
                f.write(f"{i} {y} {ap} {float(s)!r}\n")
