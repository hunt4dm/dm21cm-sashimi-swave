#!/usr/bin/env python3
"""
Generate the common 80/160 astrophysical nuisance lightcones needed by the
s-wave photon and electron Fisher forecasts.

Final directory
---------------
/work/src/dm21cm_stack/cache/p21c/swave_fisher_deltaA_split_80_160/
├── nuisance_bkg_80_160/
│   ├── LightCone_z5.0_HIIDIM=80_BOXLEN=160_fisher_fid_r12345.h5   -> symlink
│   ├── LightCone_z5.0_HIIDIM=80_BOXLEN=160_fisher_F_STAR10_-0.03_r12345.h5
│   ├── LightCone_z5.0_HIIDIM=80_BOXLEN=160_fisher_F_STAR10_0.03_r12345.h5
│   ├── ...
│   ├── nuisance_run_mapping_80_160.csv
│   └── run_configs/
│       ├── 01_F_STAR10_minus.json
│       └── ...
├── phot_delta_gpu/
└── elec_delta_gpu/

The filename convention is the same one expected by the reference
4_fisher_forecast.ipynb / py21cmfish.Parameter, with HII_DIM, BOX_LEN and seed
changed to match the existing s-wave simulations.

Important storage behavior
--------------------------
* A temporary run first produces <run_dir>/lightcones.h5.
* After successful completion it is MOVED to the final Fisher filename in
  nuisance_bkg_80_160/.
* The tiny run_config.json is archived under run_configs/.
* The remaining temporary run directory/cache is then deleted completely.
* The common fiducial is a symbolic link to the already-existing photon
  no-exotic baseline, so it occupies essentially no additional space.

Physics mapping
---------------
The 24 variations are constructed explicitly as
    12 nuisance parameters x [negative, positive].
The sequential index 1..24 is only a convenient execution label.

Usage
-----
List all variations:
    python swave_nuisance_bkg_80_160_common_final.py --list

Run all missing variations:
    python swave_nuisance_bkg_80_160_common_final.py

Run one variation:
    python swave_nuisance_bkg_80_160_common_final.py -i 1

Keep temporary run cache for debugging:
    python swave_nuisance_bkg_80_160_common_final.py --keep-cache
"""

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18

import py21cmfast as p21c
from dm21cm.evolve import evolve


# =============================================================================
# Match the existing s-wave 80/160 runs
# =============================================================================

HII_DIM = 80
BOX_LEN = 160
N_THREADS = 8
RANDOM_SEED = 12345

Z_START = 45.0
Z_END = 5.0
SUBCYCLE_FACTOR = 10
MAX_N_SHELL = None

assert np.isclose(BOX_LEN / HII_DIM, 2.0)


# =============================================================================
# Same nuisance set and shifts as the official Fisher production
# =============================================================================

ASTRO_PARAM_NAMES = [
    "F_STAR10",
    "F_STAR7_MINI",
    "ALPHA_STAR",
    "ALPHA_STAR_MINI",
    "t_STAR",
    "F_ESC10",
    "F_ESC7_MINI",
    "ALPHA_ESC",
    "L_X",
    "L_X_MINI",
    "NU_X_THRESH",
    "A_LW",
]

ASTRO_PARAM_VALUES = [
    -1.25,
    -2.5,
    0.5,
    0.0,
    0.5,
    -1.35,
    -1.35,
    -0.3,
    40.5,
    40.5,
    500,
    2.0,
]

ASTRO_PARAM_SHIFTS = [
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
    0.001,
    0.001,
    0.03,
    0.03,
]

ASTRO_FIDUCIAL = dict(zip(ASTRO_PARAM_NAMES, ASTRO_PARAM_VALUES))


# =============================================================================
# Paths
# =============================================================================

P21C_BASE_DIR = Path(
    "/work/src/dm21cm_stack/cache/p21c/swave_fisher_deltaA_split_80_160"
).expanduser().resolve()

PHOT_ROOT = P21C_BASE_DIR / "phot_delta_gpu"
ELEC_ROOT = P21C_BASE_DIR / "elec_delta_gpu"

COMMON_BKG_ROOT = P21C_BASE_DIR / "nuisance_bkg_80_160"
RUN_CONFIG_DIR = COMMON_BKG_ROOT / "run_configs"

PHOT_BASELINE_RUN_NAME = "swave_psstep_phot_gpu_noexotic_baseline"
PHOT_BASELINE_LIGHTCONE = PHOT_ROOT / PHOT_BASELINE_RUN_NAME / "lightcones.h5"

RUN_NAME_PREFIX = "swave_nuisance_bkg_"
MAPPING_CSV = COMMON_BKG_ROOT / "nuisance_run_mapping_80_160.csv"

COMMON_BKG_ROOT.mkdir(parents=True, exist_ok=True)
RUN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Temporary nuisance run directories are direct children of this root.
os.environ["P21C_CACHE_DIR"] = str(COMMON_BKG_ROOT)


# =============================================================================
# Explicit 12 x 2 nuisance mapping
# =============================================================================

def all_variations():
    rows = []
    seq = 1

    for param_index, name in enumerate(ASTRO_PARAM_NAMES):
        fid = float(ASTRO_PARAM_VALUES[param_index])

        for shift_index, direction in enumerate((-1, +1)):
            astro = dict(ASTRO_FIDUCIAL)

            if name == "ALPHA_STAR_MINI":
                # Official special case because its fiducial value is zero.
                run_value = -0.1 if direction < 0 else +0.1
                physical_shift = run_value
                filename_shift = run_value
            else:
                frac = float(ASTRO_PARAM_SHIFTS[param_index]) * direction
                run_value = fid * (1.0 + frac)

                # Reference Fisher naming stores the fractional shift in the
                # filename, not the absolute parameter displacement.
                physical_shift = frac
                filename_shift = frac

            astro[name] = float(run_value)

            rows.append(
                {
                    "seq": seq,
                    "param_index": param_index,
                    "shift_index": shift_index,
                    "direction": direction,
                    "name": name,
                    "shift": float(physical_shift),
                    "filename_shift": float(filename_shift),
                    "fiducial_value": fid,
                    "run_value": float(run_value),
                    "astro": astro,
                }
            )
            seq += 1

    assert len(rows) == 24
    return rows


VARIATIONS = all_variations()


def get_variation(seq):
    if seq < 1 or seq > 24:
        raise ValueError("Sequential nuisance index must be in [1, 24].")
    return VARIATIONS[seq - 1]


def run_name(v):
    side = "minus" if v["direction"] < 0 else "plus"
    return f"{RUN_NAME_PREFIX}{v['seq']:02d}_{v['name']}_{side}"


# =============================================================================
# Fisher-compatible filenames
# =============================================================================

def fiducial_filename():
    return (
        f"LightCone_z{Z_END:.1f}_"
        f"HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}_"
        f"fisher_fid_r{RANDOM_SEED}.h5"
    )


def output_filename(v):
    return (
        f"LightCone_z{Z_END:.1f}_"
        f"HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}_"
        f"fisher_{v['name']}_{v['filename_shift']:g}_"
        f"r{RANDOM_SEED}.h5"
    )


# =============================================================================
# Mapping / metadata
# =============================================================================

def write_mapping_csv():
    rows = []
    for v in VARIATIONS:
        rows.append(
            {
                "seq": v["seq"],
                "parameter": v["name"],
                "direction": "-" if v["direction"] < 0 else "+",
                "filename_shift": v["filename_shift"],
                "fiducial_value": v["fiducial_value"],
                "run_value": v["run_value"],
                "run_name": run_name(v),
                "final_lightcone": output_filename(v),
            }
        )

    with MAPPING_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_mapping():
    print("\nExplicit nuisance mapping")
    print("-" * 112)

    print(f" 0: fiducial -> {fiducial_filename()}")

    for v in VARIATIONS:
        side = "-" if v["direction"] < 0 else "+"
        print(
            f"{v['seq']:2d}: "
            f"{v['name']:16s} {side}  "
            f"fid={v['fiducial_value']:<10g} "
            f"run={v['run_value']:<12g} "
            f"file_shift={v['filename_shift']:<9g}"
        )


def write_run_config(run_dir, v):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "common 80/160 astrophysical nuisance Fisher lightcone",
        "run_name": run_name(v),
        "sequential_label": v["seq"],
        "parameter": v["name"],
        "direction": v["direction"],
        "filename_shift": v["filename_shift"],
        "fiducial_value": v["fiducial_value"],
        "run_value": v["run_value"],
        "simulation": {
            "HII_DIM": HII_DIM,
            "BOX_LEN": BOX_LEN,
            "N_THREADS": N_THREADS,
            "RANDOM_SEED": RANDOM_SEED,
            "z_start": Z_START,
            "z_end": Z_END,
            "subcycle_factor": SUBCYCLE_FACTOR,
            "max_n_shell": MAX_N_SHELL,
        },
        "astro_params": v["astro"],
        "paths": {
            "P21C_BASE_DIR": str(P21C_BASE_DIR),
            "COMMON_BKG_ROOT": str(COMMON_BKG_ROOT),
            "photon_baseline": str(PHOT_BASELINE_LIGHTCONE),
            "final_lightcone": str(COMMON_BKG_ROOT / output_filename(v)),
        },
    }

    live_config = run_dir / "run_config.json"
    with live_config.open("w") as f:
        json.dump(payload, f, indent=2)

    archive_config = RUN_CONFIG_DIR / (
        f"{v['seq']:02d}_{v['name']}_"
        f"{'minus' if v['direction'] < 0 else 'plus'}.json"
    )
    with archive_config.open("w") as f:
        json.dump(payload, f, indent=2)

    return live_config, archive_config


# =============================================================================
# Common fiducial: symlink, not copy
# =============================================================================

def ensure_common_fiducial():
    if not PHOT_BASELINE_LIGHTCONE.exists():
        raise FileNotFoundError(
            "Existing photon no-exotic baseline not found:\n"
            f"  {PHOT_BASELINE_LIGHTCONE}"
        )

    target = COMMON_BKG_ROOT / fiducial_filename()

    if target.is_symlink():
        current = target.resolve()
        expected = PHOT_BASELINE_LIGHTCONE.resolve()
        if current != expected:
            raise RuntimeError(
                "Existing fiducial symlink points to the wrong file:\n"
                f"  link     = {target}\n"
                f"  current  = {current}\n"
                f"  expected = {expected}"
            )
        print("[fiducial symlink exists]", target)
        return target

    if target.exists():
        raise RuntimeError(
            "A real file already exists at the intended fiducial symlink path:\n"
            f"  {target}\n"
            "Remove/rename it manually if you want this script to create a symlink."
        )

    # Use an absolute symlink so it remains valid regardless of the working dir.
    target.symlink_to(PHOT_BASELINE_LIGHTCONE.resolve())
    print("[created fiducial symlink]")
    print("  link  :", target)
    print("  target:", PHOT_BASELINE_LIGHTCONE.resolve())
    return target


# =============================================================================
# Initial conditions
# =============================================================================

def build_initial_conditions():
    p21c.global_params.CLUMPING_FACTOR = 1.0

    return p21c.initial_conditions(
        user_params=p21c.UserParams(
            HII_DIM=HII_DIM,
            BOX_LEN=BOX_LEN,
            N_THREADS=N_THREADS,
        ),
        cosmo_params=p21c.CosmoParams(
            OMm=Planck18.Om0,
            OMb=Planck18.Ob0,
            POWER_INDEX=Planck18.meta["n"],
            SIGMA_8=Planck18.meta["sigma8"],
            hlittle=Planck18.h,
        ),
        random_seed=RANDOM_SEED,
        write=True,
    )


# =============================================================================
# Cleanup
# =============================================================================

def assert_safe_temp_run_dir(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    root = COMMON_BKG_ROOT.resolve()

    forbidden = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        P21C_BASE_DIR.resolve(),
        PHOT_ROOT.resolve(),
        ELEC_ROOT.resolve(),
        root,
        RUN_CONFIG_DIR.resolve(),
    }

    data_root_env = os.environ.get("DM21CM_DATA_DIR")
    if data_root_env:
        forbidden.add(Path(data_root_env).expanduser().resolve())

    if run_dir in forbidden:
        raise RuntimeError(f"Refusing to delete protected path: {run_dir}")

    if run_dir.parent != root:
        raise RuntimeError(
            "Refusing to delete a directory that is not a direct child of "
            "COMMON_BKG_ROOT:\n"
            f"  run_dir = {run_dir}\n"
            f"  root    = {root}"
        )

    if not run_dir.name.startswith(RUN_NAME_PREFIX):
        raise RuntimeError(
            "Refusing to delete directory with unexpected name:\n"
            f"  {run_dir}"
        )

    return run_dir


def remove_temp_run_dir(run_dir):
    run_dir = assert_safe_temp_run_dir(run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print("[cleanup] removed temporary run directory:", run_dir)


# =============================================================================
# One nuisance variation
# =============================================================================

def run_one(seq, keep_cache=False):
    v = get_variation(seq)

    this_run_name = run_name(v)
    run_dir = COMMON_BKG_ROOT / this_run_name
    generated = run_dir / "lightcones.h5"
    final_file = COMMON_BKG_ROOT / output_filename(v)

    # Completed final result already exists.
    if final_file.exists():
        print(f"\n[skip completed] {seq:02d} {v['name']}")
        print(" ", final_file)

        # Clean stale temporary cache from an earlier successful run.
        if run_dir.exists() and not keep_cache:
            remove_temp_run_dir(run_dir)
        return final_file

    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(run_dir, v)

    # Recovery: if evolution finished before but final rename did not happen,
    # move the already-complete lightcone now instead of rerunning.
    if generated.exists():
        print(f"\n[recover completed temporary lightcone] {seq:02d}")
        print("  from:", generated)
        print("  to  :", final_file)
        shutil.move(str(generated), str(final_file))

        if not keep_cache:
            remove_temp_run_dir(run_dir)
        return final_file

    print("\n" + "=" * 104)
    print("Sequential label :", seq)
    print("Parameter        :", v["name"])
    print("Direction        :", "-" if v["direction"] < 0 else "+")
    print("Filename shift   :", v["filename_shift"])
    print("Fiducial value   :", v["fiducial_value"])
    print("Run value        :", v["run_value"])
    print("Run name         :", this_run_name)
    print("Temporary dir    :", run_dir)
    print("Final lightcone  :", final_file)
    print("=" * 104)

    os.environ["P21C_CACHE_DIR"] = str(COMMON_BKG_ROOT)
    p21c.config["direc"] = str(run_dir)

    p21c_initial_conditions = build_initial_conditions()
    p21c_astro_params = p21c.AstroParams(**v["astro"])

    # Guard against accidental AstroParams mismatch.
    for name, expected in v["astro"].items():
        got = float(getattr(p21c_astro_params, name))
        if not np.isclose(got, float(expected)):
            raise RuntimeError(
                f"AstroParams mismatch for {name}: got {got}, expected {expected}"
            )

    result = evolve(
        run_name=this_run_name,
        z_start=Z_START,
        z_end=Z_END,
        subcycle_factor=SUBCYCLE_FACTOR,
        max_n_shell=MAX_N_SHELL,
        resume=False,
        use_tqdm=True,
        injection=None,
        p21c_initial_conditions=p21c_initial_conditions,
        p21c_astro_params=p21c_astro_params,
        use_DH_init=True,
        rerun_DH=False,
        homogenize_injection=False,
        homogenize_deposition=False,
    )

    # Normal DM21cm behavior is to write run_dir/lightcones.h5.
    if not generated.exists():
        lc = result.get("lightcone") if isinstance(result, dict) else None
        if lc is None:
            raise FileNotFoundError(
                "Evolution finished but no lightcones.h5 was found and no "
                "lightcone object was returned:\n"
                f"  {run_dir}"
            )
        lc._write(
            fname="lightcones.h5",
            direc=str(run_dir),
            clobber=True,
        )

    if not generated.exists():
        raise FileNotFoundError(
            f"Expected generated lightcone does not exist: {generated}"
        )

    # Critical order:
    # 1. MOVE the completed lightcone to its Fisher-compatible final filename.
    # 2. Verify the destination exists.
    # 3. Only then delete the temporary cache directory.
    shutil.move(str(generated), str(final_file))

    if not final_file.exists():
        raise FileNotFoundError(
            f"Move failed; final lightcone was not created: {final_file}"
        )

    print("\n[moved Fisher-ready lightcone]")
    print(" ", final_file)

    if keep_cache:
        print("[keep-cache] temporary directory retained:", run_dir)
    else:
        remove_temp_run_dir(run_dir)

    return final_file


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate common 80/160 astrophysical nuisance lightcones for "
            "s-wave photon/electron Fisher forecasts."
        )
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=None,
        help="Sequential nuisance label 1..24. Omit to run all missing variations.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the explicit 12 x 2 nuisance mapping and exit.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not delete temporary DM21cm/21cmFAST run directories after success.",
    )
    args = parser.parse_args()

    write_mapping_csv()

    print("P21C_BASE_DIR    =", P21C_BASE_DIR)
    print("COMMON_BKG_ROOT  =", COMMON_BKG_ROOT)
    print("MAPPING_CSV      =", MAPPING_CSV)
    print("PHOT_BASELINE    =", PHOT_BASELINE_LIGHTCONE)
    print("RUN_CONFIG_DIR   =", RUN_CONFIG_DIR)

    if args.list:
        print_mapping()
        return

    ensure_common_fiducial()

    if args.index is not None:
        run_one(args.index, keep_cache=args.keep_cache)
    else:
        for v in VARIATIONS:
            run_one(v["seq"], keep_cache=args.keep_cache)

    print("\nAll requested nuisance lightcones are complete.")
    print("Use this directory directly as py21cmfish lightcone_dir / BKG_DIR:")
    print(" ", COMMON_BKG_ROOT)


if __name__ == "__main__":
    main()
