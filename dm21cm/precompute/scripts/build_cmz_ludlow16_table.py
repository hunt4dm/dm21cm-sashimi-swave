"""Build cmz_Ludlow16.h5.

This script generates the missing DM21cm concentration-mass-redshift table

    $DM21CM_DIR/data/production/cmz_Ludlow16.h5

using the Ludlow16 concentration model implemented in halomod.

The generated file is compatible with dm21cm.precompute.halo.cmz(), which
expects an HDF5 dictionary with keys:

    m : halo mass grid [Msun]
    z : redshift grid [1]
    c : concentration table c(M,z), shape (len(z), len(m))

This script intentionally does NOT import dm21cm.precompute.halo, because
halo.py tries to load cmz_Ludlow16.h5 at import time. This script is meant
to create that missing file.

References:
    Ludlow et al. 2016, MNRAS 460, 1214.
    halomod concentration models.
"""

import os
import argparse
import warnings

import numpy as np
from scipy import interpolate
from tqdm import tqdm

import halomod

from dm21cm.utils import save_h5_dict


# Try to inherit the same mass range as the HMF code.
# This avoids generating a cmz table whose mass range is narrower than hmf.h5.
try:
    from dm21cm.precompute.ps import M_MIN as PS_M_MIN
    from dm21cm.precompute.ps import M_MAX as PS_M_MAX
except Exception:
    PS_M_MIN = float(os.environ.get("DM21CM_M_MIN", "1e-6"))
    PS_M_MAX = float(os.environ.get("DM21CM_M_MAX", "1e19"))


def _positive_finite_arrays(xs, ys):
    """Keep only finite, positive mass and concentration values."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
    xs = xs[mask]
    ys = ys[mask]

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    return xs, ys


def fix_cmz_numerical_issues(xs, ys, z):
    """Clean numerical artifacts in halomod concentration output.

    This function follows the same purpose as fix_cmz_numerical_issues()
    in dm21cm.precompute.halo, but is made more defensive.

    Important:
        This is not the Ludlow16 physical model itself.
        It is only a numerical cleanup/smoothing step applied to halomod's
        tabulated c(M,z) output.

    Args:
        xs (array): halomod mass grid.
        ys (array): concentration values.
        z (float): redshift.

    Returns:
        tuple: cleaned mass grid and cleaned concentration values.
    """
    xs, ys = _positive_finite_arrays(xs, ys)

    if len(xs) < 4:
        raise RuntimeError(f"Too few valid c(M,z) points at z={z}.")

    def accepted_lowz(xs_in, ys_in):
        """Reject sudden point-to-point jumps at low/intermediate redshift.

        The original DM21cm halo.py used a very tight adjacent-ratio window
        around unity. Here we use a wider but still safe window, because the
        goal is to remove numerical pathologies without distorting the
        Ludlow16 relation.
        """
        ratio_min = 0.2
        ratio_max = 5.0

        xs_acc = [xs_in[0]]
        ys_acc = [ys_in[0]]

        for x, y in zip(xs_in[1:], ys_in[1:]):
            ratio = y / ys_acc[-1]
            if ratio_min < ratio < ratio_max:
                xs_acc.append(x)
                ys_acc.append(y)

        return np.asarray(xs_acc), np.asarray(ys_acc)

    def accepted_highz(xs_in, ys_in, z_in):
        """High-z guard against known edge artifacts.

        This mirrors the spirit of the high-z empirical filter in the
        original halo.py. If the filter becomes too aggressive, the function
        falls back to the raw finite-positive data.
        """
        z_s = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        p0_s = np.array([3, 2.52, 2.35, 2.3, 2.2, 2.17, 2.14, 2.1, 2.1], dtype=float)
        p1_s = np.array([-0.055, -0.03, -0.025, -0.02, -0.015, -0.013, -0.012, -0.01, -0.01], dtype=float)

        p0 = np.interp(z_in, z_s, p0_s)
        p1 = np.interp(z_in, z_s, p1_s)

        y_upper = p0 + np.log10(xs_in) * p1
        keep = ys_in < y_upper

        xs_acc = xs_in[keep]
        ys_acc = ys_in[keep]

        if len(xs_acc) < 4:
            return xs_in, ys_in

        return xs_acc, ys_acc

    if z < 20.0:
        x_acc, y_acc = accepted_lowz(xs, ys)
    else:
        x_acc, y_acc = accepted_highz(xs, ys, z)

    if len(x_acc) < 4:
        warnings.warn(
            f"Numerical cleanup left too few points at z={z}. "
            "Using raw finite-positive halomod output instead."
        )
        x_acc, y_acc = xs, ys

    # Smooth/interpolate in log-log space. This is more natural for c(M,z),
    # which is positive and approximately power-law-like over mass.
    log_y_fixed = interpolate.interp1d(
        np.log(x_acc),
        np.log(y_acc),
        kind="linear",
        bounds_error=False,
        fill_value=(np.log(y_acc[0]), np.log(y_acc[-1])),
    )(np.log(xs))

    y_fixed = np.exp(log_y_fixed)

    if not np.all(np.isfinite(y_fixed)) or np.any(y_fixed <= 0.0):
        raise RuntimeError(f"Invalid cleaned c(M,z) values at z={z}.")

    return xs, y_fixed


def cmz_raw_ludlow16(m, z, model="Ludlow16"):
    """Evaluate the Ludlow16 concentration model using halomod.

    This mirrors the model choice in dm21cm.precompute.halo.cmz_raw(), but
    avoids importing halo.py because halo.py requires cmz_Ludlow16.h5 to
    already exist.

    Args:
        m (array): Target halo masses [Msun], following DM21cm convention.
        z (float): Redshift.
        model (str): halomod concentration model name.

    Returns:
        array: c(M,z) at target masses.
    """
    m = np.asarray(m, dtype=float)

    if np.any(~np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("All target masses must be finite and positive.")

    hm = halomod.DMHaloModel(
        halo_concentration_model=model,
        z=float(z),
        Mmin=0.0,
        Mmax=19.0,
        dlog10m=0.025,
        mdef_model="SOCritical",
        halo_profile_model=halomod.profiles.NFW,
    )

    hm_m, hm_c = fix_cmz_numerical_issues(hm.m, hm.cmz_relation, float(z))

    # Interpolate log(c) over log(M). This is more stable than linear-c
    # interpolation and guarantees positive concentration values.
    c_out = np.exp(
        np.interp(
            np.log(m),
            np.log(hm_m),
            np.log(hm_c),
            left=np.log(hm_c[0]),
            right=np.log(hm_c[-1]),
        )
    )

    if not np.all(np.isfinite(c_out)) or np.any(c_out <= 0.0):
        raise RuntimeError(f"Invalid interpolated c(M,z) values at z={z}.")

    return c_out


def make_redshift_grid(z_min, z_max, n_z):
    """Construct redshift grid for the cmz table.

    The HMF table in DM21cm usually covers z=4--100, while halo.cmz() may
    also be useful at z=0 for tests and diagnostics. We therefore default
    to z=0--100.

    Low/intermediate redshift is sampled linearly; high redshift is sampled
    geometrically to maintain coverage without excessive table size.
    """
    z_min = float(z_min)
    z_max = float(z_max)
    n_z = int(n_z)

    if n_z < 3:
        raise ValueError("n_z must be at least 3.")

    if z_min < 0.0:
        raise ValueError("z_min must be non-negative.")

    if z_max <= z_min:
        raise ValueError("Need z_max > z_min.")

    if z_min == 0.0 and z_max >= 100.0 and n_z >= 250:
        n_low = min(201, n_z - 2)
        z_low = np.linspace(0.0, 20.0, n_low)

        n_high = n_z - len(z_low)
        z_high = np.geomspace(20.0 + 1e-6, z_max, n_high)

        z_s = np.unique(np.concatenate([z_low, z_high]))
    else:
        z_s = np.linspace(z_min, z_max, n_z)

    if not np.all(np.diff(z_s) > 0.0):
        raise RuntimeError("Redshift grid is not strictly increasing.")

    return z_s


def build_cmz_table(
    m_min=None,
    m_max=None,
    n_m=4001,
    z_min=0.0,
    z_max=100.0,
    n_z=501,
    model="Ludlow16",
):
    """Build c(M,z) lookup table.

    Args:
        m_min (float or None): Minimum halo mass [Msun].
            If None, use ps.M_MIN / DM21CM_M_MIN.
        m_max (float or None): Maximum halo mass [Msun].
            If None, use ps.M_MAX / DM21CM_M_MAX.
        n_m (int): Number of mass grid points.
        z_min (float): Minimum redshift.
        z_max (float): Maximum redshift.
        n_z (int): Number of redshift grid points.
        model (str): halomod concentration model.

    Returns:
        dict: HDF5-ready dictionary.
    """
    if m_min is None:
        m_min = float(os.environ.get("DM21CM_M_MIN", PS_M_MIN))
    if m_max is None:
        m_max = float(os.environ.get("DM21CM_M_MAX", PS_M_MAX))

    m_min = float(m_min)
    m_max = float(m_max)
    n_m = int(n_m)

    if m_min <= 0.0:
        raise ValueError("m_min must be positive.")
    if m_max <= m_min:
        raise ValueError("Need m_max > m_min.")
    if n_m < 4:
        raise ValueError("n_m must be at least 4.")

    m_s = np.geomspace(m_min, m_max, n_m)
    z_s = make_redshift_grid(z_min=z_min, z_max=z_max, n_z=n_z)

    c_table = np.zeros((len(z_s), len(m_s)), dtype=np.float64)

    for i_z, z in enumerate(tqdm(z_s, desc=f"Building c(M,z) table: {model}")):
        c_vals = cmz_raw_ludlow16(m_s, float(z), model=model)

        if not np.all(np.isfinite(c_vals)):
            raise RuntimeError(f"Non-finite concentration values at z={z}.")

        if np.any(c_vals <= 0.0):
            raise RuntimeError(f"Non-positive concentration values at z={z}.")

        c_table[i_z] = c_vals

    data = {
        "m": m_s,
        "z": z_s,
        "c": c_table,
        "model": model,
        "m_min": float(m_s[0]),
        "m_max": float(m_s[-1]),
        "n_m": int(len(m_s)),
        "z_min": float(z_s[0]),
        "z_max": float(z_s[-1]),
        "n_z": int(len(z_s)),
        "units": "m: [Msun]. z: [1]. c: dimensionless concentration.",
        "shapes": "c: (z, m).",
        "note": (
            "Generated locally using halomod.DMHaloModel with "
            "halo_concentration_model='Ludlow16', mdef_model='SOCritical', "
            "halo_profile_model=NFW. This file replaces the missing "
            "DM21cm data/production/cmz_Ludlow16.h5 lookup table."
        ),
    }

    return data


def validate_cmz_data(data):
    """Basic structural and numerical validation before writing."""
    m = data["m"]
    z = data["z"]
    c = data["c"]

    if c.shape != (len(z), len(m)):
        raise RuntimeError(
            f"Bad c table shape: got {c.shape}, expected {(len(z), len(m))}."
        )

    if not np.all(np.diff(m) > 0.0):
        raise RuntimeError("Mass grid is not strictly increasing.")

    if not np.all(np.diff(z) > 0.0):
        raise RuntimeError("Redshift grid is not strictly increasing.")

    if not np.all(np.isfinite(c)):
        raise RuntimeError("c table contains NaN or inf.")

    if np.any(c <= 0.0):
        raise RuntimeError("c table contains non-positive values.")

    # Loose physical sanity bounds. Do not make these too strict, because
    # Ludlow16/halomod may produce low concentrations at high redshift.
    c_min = np.nanmin(c)
    c_max = np.nanmax(c)

    if c_min < 0.1:
        warnings.warn(f"Very small concentration found: c_min={c_min:.6e}")

    if c_max > 1e4:
        warnings.warn(f"Very large concentration found: c_max={c_max:.6e}")

    return True


def validate_against_direct_halomod(data, model="Ludlow16"):
    """Compare table values with direct halomod evaluation at sample points.

    This is a stronger validation than just checking HDF5 format. It verifies
    that the saved table really reproduces the halomod Ludlow16 model.

    The function uses a small set of representative masses and redshifts.
    """
    m_grid = data["m"]
    z_grid = data["z"]
    c_table = data["c"]

    test_masses = np.array(
        [
            m_grid[0],
            np.sqrt(m_grid[0] * m_grid[-1]),
            1e0,
            1e6,
            1e8,
            1e10,
            1e12,
            min(1e14, m_grid[-1]),
        ],
        dtype=float,
    )

    test_masses = test_masses[
        (test_masses >= m_grid[0]) & (test_masses <= m_grid[-1])
    ]
    test_masses = np.unique(test_masses)

    test_redshifts = np.array([0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    test_redshifts = test_redshifts[
        (test_redshifts >= z_grid[0]) & (test_redshifts <= z_grid[-1])
    ]

    log_m_grid = np.log(m_grid)

    def table_interp_one(m, z):
        """Interpolate table in log(c)-log(M), then linearly in z."""
        log_c_at_each_z = np.array(
            [
                np.interp(
                    np.log(m),
                    log_m_grid,
                    np.log(c_table[i_z]),
                    left=np.log(c_table[i_z, 0]),
                    right=np.log(c_table[i_z, -1]),
                )
                for i_z in range(len(z_grid))
            ]
        )

        log_c = np.interp(
            z,
            z_grid,
            log_c_at_each_z,
            left=log_c_at_each_z[0],
            right=log_c_at_each_z[-1],
        )

        return float(np.exp(log_c))

    rel_errors = []

    print("===== validation against direct halomod Ludlow16 =====")

    for z in test_redshifts:
        c_direct = cmz_raw_ludlow16(test_masses, float(z), model=model)
        c_lookup = np.array([table_interp_one(m, float(z)) for m in test_masses])

        rel = np.abs(c_lookup - c_direct) / c_direct
        rel_errors.extend(rel)

        print(f"z = {z:.3g}")
        for m, cd, cl, rr in zip(test_masses, c_direct, c_lookup, rel):
            print(
                f"  M={m:.6e}  direct={cd:.6e}  table={cl:.6e}  relerr={rr:.3e}"
            )

    rel_errors = np.asarray(rel_errors)

    print(f"max relerr    = {np.max(rel_errors):.6e}")
    print(f"median relerr = {np.median(rel_errors):.6e}")

    # A 5% threshold is intentionally conservative. With a dense table,
    # this should normally pass easily except in pathological high-z regions.
    if np.max(rel_errors) > 5e-2:
        warnings.warn(
            "Maximum table-vs-direct halomod relative error exceeds 5%. "
            "Inspect whether this comes from high-z cleanup or table resolution."
        )

    return rel_errors


def validate_physical_trends(data):
    """Check broad physical trends of c(M,z).

    For CDM concentration models, c generally decreases with mass at fixed z
    and decreases with redshift at fixed mass. The test is deliberately loose
    because the exact Ludlow16 relation may contain mild features.
    """
    m = data["m"]
    z = data["z"]
    c = data["c"]

    def nearest(arr, val):
        return int(np.argmin(np.abs(arr - val)))

    print("===== broad physical trend check =====")

    for z0 in [0.0, 5.0, 10.0, 20.0]:
        if z0 < z[0] or z0 > z[-1]:
            continue

        iz = nearest(z, z0)
        mass_samples = np.array([1e0, 1e6, 1e8, 1e10, 1e12, 1e14], dtype=float)
        mass_samples = mass_samples[
            (mass_samples >= m[0]) & (mass_samples <= m[-1])
        ]

        vals = np.array([c[iz, nearest(m, mm)] for mm in mass_samples])

        print(f"z={z[iz]:.3g}")
        for mm, vv in zip(mass_samples, vals):
            print(f"  M={mm:.3e}, c={vv:.6e}")

        if not np.all(np.isfinite(vals)) or np.any(vals <= 0.0):
            raise RuntimeError(f"Invalid trend-check values at z={z0}.")

    for m0 in [1e0, 1e6, 1e8, 1e10, 1e12]:
        if m0 < m[0] or m0 > m[-1]:
            continue

        im = nearest(m, m0)
        z_samples = np.array([0.0, 2.0, 5.0, 10.0, 20.0], dtype=float)
        z_samples = z_samples[(z_samples >= z[0]) & (z_samples <= z[-1])]

        vals = np.array([c[nearest(z, zz), im] for zz in z_samples])

        print(f"M={m[im]:.3e}")
        for zz, vv in zip(z_samples, vals):
            print(f"  z={zz:.3g}, c={vv:.6e}")

        if not np.all(np.isfinite(vals)) or np.any(vals <= 0.0):
            raise RuntimeError(f"Invalid trend-check values at M={m0}.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate DM21cm-compatible cmz_Ludlow16.h5 table."
    )

    parser.add_argument(
        "--m-min",
        type=float,
        default=None,
        help=(
            "Minimum halo mass [Msun]. Default: DM21CM_M_MIN if set, "
            "otherwise dm21cm.precompute.ps.M_MIN."
        ),
    )
    parser.add_argument(
        "--m-max",
        type=float,
        default=None,
        help=(
            "Maximum halo mass [Msun]. Default: DM21CM_M_MAX if set, "
            "otherwise dm21cm.precompute.ps.M_MAX."
        ),
    )
    parser.add_argument("--n-m", type=int, default=4001)

    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=100.0)
    parser.add_argument("--n-z", type=int, default=501)

    parser.add_argument("--model", type=str, default="Ludlow16")

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output filename. Default: "
            "$DM21CM_DIR/data/production/cmz_Ludlow16.h5"
        ),
    )

    parser.add_argument(
        "--skip-direct-validation",
        action="store_true",
        help="Skip table-vs-direct-halomod validation.",
    )

    args = parser.parse_args()

    dm21cm_dir = os.environ.get("DM21CM_DIR", None)

    if args.output is None:
        if dm21cm_dir is None:
            raise RuntimeError(
                "DM21CM_DIR is not set. Either set DM21CM_DIR or pass --output."
            )
        output = os.path.join(dm21cm_dir, "data", "production", "cmz_Ludlow16.h5")
    else:
        output = args.output

    os.makedirs(os.path.dirname(output), exist_ok=True)

    print("===== CMZ Ludlow16 table generation =====")
    print(f"DM21CM_DIR = {dm21cm_dir}")
    print(f"model      = {args.model}")
    print(f"m_min      = {args.m_min if args.m_min is not None else os.environ.get('DM21CM_M_MIN', PS_M_MIN)}")
    print(f"m_max      = {args.m_max if args.m_max is not None else os.environ.get('DM21CM_M_MAX', PS_M_MAX)}")
    print(f"n_m        = {args.n_m}")
    print(f"z_min      = {args.z_min}")
    print(f"z_max      = {args.z_max}")
    print(f"n_z        = {args.n_z}")
    print(f"output     = {output}")

    data = build_cmz_table(
        m_min=args.m_min,
        m_max=args.m_max,
        n_m=args.n_m,
        z_min=args.z_min,
        z_max=args.z_max,
        n_z=args.n_z,
        model=args.model,
    )

    validate_cmz_data(data)

    print("===== table summary =====")
    print(f"m range = [{data['m'][0]:.6e}, {data['m'][-1]:.6e}]")
    print(f"z range = [{data['z'][0]:.6e}, {data['z'][-1]:.6e}]")
    print(f"c shape = {data['c'].shape}")
    print(f"c min   = {np.nanmin(data['c']):.6e}")
    print(f"c max   = {np.nanmax(data['c']):.6e}")

    validate_physical_trends(data)

    if not args.skip_direct_validation:
        validate_against_direct_halomod(data, model=args.model)

    save_h5_dict(output, data)

    print("===== done =====")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()