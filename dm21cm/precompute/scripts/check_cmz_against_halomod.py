import os
import numpy as np
import h5py

from build_cmz_ludlow16_table import cmz_raw_ludlow16


def interp_table(m_query, z_query, m_grid, z_grid, c_table):
    # bilinear interpolation in z and logM, with logc interpolation in mass
    logm_grid = np.log(m_grid)
    logm_q = np.log(m_query)

    # first interpolate in mass for each z slice
    c_at_z = np.array([
        np.exp(np.interp(logm_q, logm_grid, np.log(c_table[i])))
        for i in range(len(z_grid))
    ])

    # then interpolate in z
    return np.interp(z_query, z_grid, c_at_z)


fn = os.environ["DM21CM_DIR"] + "/data/production/cmz_Ludlow16.h5"

with h5py.File(fn, "r") as f:
    m_grid = f["m"][:]
    z_grid = f["z"][:]
    c_table = f["c"][:]

test_masses = np.array([1e-6, 1e0, 1e6, 1e8, 1e10, 1e12, 1e14])
test_redshifts = np.array([0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0])

errs = []

for z in test_redshifts:
    c_direct = cmz_raw_ludlow16(test_masses, z, model="Ludlow16")
    c_lookup = np.array([
        interp_table(m, z, m_grid, z_grid, c_table)
        for m in test_masses
    ])

    rel = np.abs(c_lookup - c_direct) / c_direct
    errs.extend(rel)

    print(f"\nz = {z}")
    for m, cd, cl, r in zip(test_masses, c_direct, c_lookup, rel):
        print(f"M={m:.2e}  direct={cd:.4g}  table={cl:.4g}  relerr={r:.3e}")

errs = np.array(errs)
print("\nmax relerr =", errs.max())
print("median relerr =", np.median(errs))

assert np.all(np.isfinite(errs))
assert errs.max() < 0.05

print("HALOMOD COMPARISON CHECK PASSED")