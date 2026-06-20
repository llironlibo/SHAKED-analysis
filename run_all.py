"""
SHAKED analysis -- master runner.

Runs the numbered analysis stages 01..16 in order, then generates the figures
(Python figures first, then the R figures via Rscript). Each stage prints a
banner; any stage that errors stops the run. The R figure step is wrapped so a
missing R interpreter warns but does not fail the core analysis run.

Usage (from the repository root):
    python run_all.py

Place the de-identified cohort at data/cleaned/unified_pilot_cohort.csv first
(see data/README.md). Outputs are written under results/ and results/figures/.
"""

import os
import sys
import glob
import shutil
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))


def banner(text, char="="):
    line = char * 80
    print(f"\n{line}\n{text}\n{line}", flush=True)


def run_python_stage(script_path):
    """Run one analysis stage; return True on success (exit 0)."""
    result = subprocess.run([sys.executable, script_path], cwd=BASE)
    return result.returncode == 0


def main():
    banner("SHAKED ANALYSIS -- FULL PIPELINE")
    overall_start = time.time()

    # Analysis stages 01..16, discovered by their numeric prefix and run in order.
    stage_scripts = sorted(
        glob.glob(os.path.join(BASE, "[0-1][0-9]_*.py"))
    )
    if not stage_scripts:
        print("ERROR: no numbered stage scripts (NN_*.py) found.")
        sys.exit(1)

    results = []
    for script in stage_scripts:
        name = os.path.basename(script)
        banner(f"RUNNING STAGE: {name}")
        start = time.time()
        ok = run_python_stage(script)
        elapsed = time.time() - start
        status = "OK" if ok else f"FAILED (exit != 0)"
        results.append((name, status, elapsed))
        print(f"\n  >>> {name}: {status} ({elapsed:.1f}s)", flush=True)
        if not ok:
            banner("PIPELINE STOPPED -- a stage failed", char="!")
            for n, s, e in results:
                print(f"  {n:<40} {s:<20} {e:>7.1f}s")
            sys.exit(1)

    # ---- Figures: Python first ------------------------------------------------
    banner("GENERATING FIGURES (Python)")
    py_figs = [
        ("figures/figure1_consort.py", []),
        ("figures/figure2_forest_dose_response.py", ["final"]),
    ]
    for rel, args in py_figs:
        path = os.path.join(BASE, rel)
        print(f"\n--- {rel} ---", flush=True)
        r = subprocess.run([sys.executable, path] + args, cwd=BASE)
        status = "OK" if r.returncode == 0 else "FAILED"
        results.append((os.path.basename(rel), status, 0.0))
        print(f"  >>> {os.path.basename(rel)}: {status}", flush=True)
        if r.returncode != 0:
            banner("PIPELINE STOPPED -- a Python figure failed", char="!")
            sys.exit(1)

    # ---- Figures: R via Rscript (optional) -----------------------------------
    banner("GENERATING FIGURES (R) -- optional")
    rscript = shutil.which("Rscript")
    if rscript is None:
        print("[WARN] Rscript not found on PATH; skipping the R figures.")
        print("       The R figure scripts under figures/ can be run manually")
        print("       once R (>= 4.5, ggplot2 >= 4.0) is installed.")
    else:
        r_figs = [
            "figures/figure2_forest_dose_response.R",
            "figures/figure3_adoption_dynamics.R",
            "figures/efigure1_propensity_balance.R",
        ]
        for rel in r_figs:
            path = os.path.join(BASE, rel)
            print(f"\n--- {rel} ---", flush=True)
            r = subprocess.run([rscript, path], cwd=BASE)
            status = "OK" if r.returncode == 0 else "WARN (non-zero exit)"
            results.append((os.path.basename(rel), status, 0.0))
            print(f"  >>> {os.path.basename(rel)}: {status}", flush=True)

    total = time.time() - overall_start
    banner("PIPELINE SUMMARY")
    print(f"\n  {'Step':<42} {'Status':<22} {'Time':>8}")
    print(f"  {'-'*72}")
    for name, status, elapsed in results:
        print(f"  {name:<42} {status:<22} {elapsed:>7.1f}s")
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f} min)")
    print("\n  ALL ANALYSIS STAGES COMPLETED.")


if __name__ == "__main__":
    main()
