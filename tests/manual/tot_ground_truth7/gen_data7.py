#!/usr/bin/env python3
"""Two-sided external variant, matching TA's passing csv_like test
structure exactly: g(i,x,y) * C(i,x,z;a') -> R(i,y,z;a').
h={i} nonempty, e={y,z} (y external-only-to-g, z external-only-to-C),
contracted={x}. This mirrors csv_like's (i2,i1,m;a)*(m,i2,K)->(i1,i2,K;a)
structure: h={i2}, e={i1(only a),K(only b)}, contracted={m}."""
pno_count = {0: 2, 1: 3}
n_x = 3
n_y = 2
n_z = 2

def gval(i, x, y):
    return 0.5 + i * 2.0 - x * 0.3 + y * 0.7

def cval(i, x, z, ap):
    return 0.1 + i * 1.0 + x * 0.05 + z * 0.2 + ap * 0.01

with open("g7.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            for y in range(n_y):
                f.write(f"{i} {x} {y} {gval(i, x, y)}\n")

with open("C7.coo", "w") as f:
    for i in range(2):
        for x in range(n_x):
            for z in range(n_z):
                for ap in range(pno_count[i]):
                    f.write(f"{i} {x} {z} {ap} {cval(i, x, z, ap)}\n")

with open("R7_reference.txt", "w") as f:
    for i in range(2):
        for y in range(n_y):
            for z in range(n_z):
                for ap in range(pno_count[i]):
                    s = 0.0
                    for x in range(n_x):
                        s += gval(i, x, y) * cval(i, x, z, ap)
                    f.write(f"{i} {y} {z} {ap} {float(s)!r}\n")
