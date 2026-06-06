"""
Heterogeneous Treatment Effects Regression

Gap_mt = alpha_m + alpha_t
       + beta1 (New_m  x Post_t)
       + beta2 (Cont_m x Post_t)
       + beta3 (Exit_m x Post_t)
       + gamma X_mt + e_mt

Group definitions
-----------------
New_m  (4 muns) : on 2018 list but NOT 2010 list
                  Helsingør (217), Guldborgsund (376), Vejle (630), Silkeborg (740)

Cont_m (11 muns): on BOTH 2010 and 2018 lists
                  Copenhagen (101), Høje-Taastrup (169), Holbæk (316), Slagelse (330),
                  Odense (461), Sønderborg (540), Esbjerg (561), Horsens (615),
                  Kolding (621), Aarhus (751), Viborg (791)

Exit_m (6 muns) : on 2010 list but NOT 2018 list  [falsification group]
                  Greve (253), Fredericia (607), Svendborg (479),
                  Thisted (787), Aalborg (851), Ishøj (183)

Control         : all remaining municipalities (never treated)

Post_t = 1 for t >= 2018, zero otherwise
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths & Data ──────────────────────────────────────────────
BASE = Path(r"d:/ONGOING COURSES/Economic Policy and the Welfare State/output")
df   = pd.read_csv(BASE / "panel_mun_year_with_ras206.csv")
df   = df.sort_values(["MUN_CODE","YEAR"]).reset_index(drop=True)

CONTROLS  = ["SH_SHARE","LOG_INCOME","NW_IMMIG_SHARE"]
POST_YEAR = 2018          # actual 2018 legislation
REF_YEAR  = 2017          # event-study reference
YEARS     = sorted(df["YEAR"].unique())

# ── Group membership (MUN_CODEs) ──────────────────────────────
NEW_CODES  = {217, 376, 630, 740}          # newly treated 2018
CONT_CODES = {101, 169, 316, 330, 461,
              540, 561, 615, 621, 751, 791} # continuously treated
EXIT_CODES = {253, 607, 479, 787, 851, 183} # exited (falsification)
# Control = everyone else (not in any of the above)

# ── Assign group labels ───────────────────────────────────────
def assign_group(code):
    if code in NEW_CODES:   return "New"
    if code in CONT_CODES:  return "Cont"
    if code in EXIT_CODES:  return "Exit"
    return "Control"

df["GROUP"] = df["MUN_CODE"].map(assign_group)

# Verify
print("=" * 65)
print("HETEROGENEOUS TREATMENT EFFECTS REGRESSION")
print("Gap_mt = a_m + a_t + b1(New×Post) + b2(Cont×Post) + b3(Exit×Post) + gX + e")
print("=" * 65)
print(f"\n  Post_t = 1 for t >= {POST_YEAR}")
print(f"\n  Group composition:")
for grp, desc in [("New","Newly treated 2018 (no prior exposure)"),
                  ("Cont","Continuously treated (2010 & 2018)"),
                  ("Exit","Exited — falsification group (2010 only)"),
                  ("Control","Never treated — pure control")]:
    muns = df[df.GROUP==grp][["MUN_CODE","MUN_NAME"]].drop_duplicates().sort_values("MUN_NAME")
    print(f"\n  {grp:8s} ({len(muns):2d} muns) — {desc}:")
    for _, row in muns.iterrows():
        print(f"    {int(row.MUN_CODE):4d}  {row.MUN_NAME}")

# ── Group dummies & Post interactions ────────────────────────
df["Post"]     = (df["YEAR"] >= POST_YEAR).astype(float)
df["NewPost"]  = (df["GROUP"] == "New").astype(float)  * df["Post"]
df["ContPost"] = (df["GROUP"] == "Cont").astype(float) * df["Post"]
df["ExitPost"] = (df["GROUP"] == "Exit").astype(float) * df["Post"]

# ── Fixed-effect dummies ──────────────────────────────────────
mun_dum  = pd.get_dummies(df["MUN_CODE"], prefix="mun",  drop_first=True).astype(float)
year_dum = pd.get_dummies(df["YEAR"],     prefix="year", drop_first=True).astype(float)

# ── Helpers ───────────────────────────────────────────────────
def stars(p):
    if p<0.001: return "***"
    if p<0.01:  return "**"
    if p<0.05:  return "*"
    if p<0.10:  return "."
    return ""

def run_ols(outcome, extra_regs):
    X = pd.concat([df[extra_regs], mun_dum, year_dum], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[outcome]
    valid = y.notna() & X.notna().all(axis=1)
    res = sm.OLS(y[valid], X[valid]).fit(
        cov_type="cluster",
        cov_kwds={"groups": df.loc[valid, "MUN_CODE"]}
    )
    return res

def print_coef(res, name, label):
    if name not in res.params.index:
        print(f"    {label:35s}: [not estimated]")
        return None, None
    c = res.params[name]; s = res.bse[name]
    t = res.tvalues[name]; p = res.pvalues[name]
    ci = res.conf_int().loc[name]
    print(f"    {label:35s}: {c:+.4f}  SE={s:.4f}  t={t:.3f}  p={p:.4f} {stars(p)}")
    print(f"    {'':35s}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    return c, p

# ══════════════════════════════════════════════════════════════
# REGRESSIONS — four specifications
# ══════════════════════════════════════════════════════════════
INTERACTION_REGS = ["NewPost", "ContPost", "ExitPost"]

specs = {
    "(A) EmpGap,   no controls":    ("EMP_GAP",   INTERACTION_REGS),
    "(B) EmpGap,   with controls":  ("EMP_GAP",   INTERACTION_REGS + CONTROLS),
    "(C) UnempGap, no controls":    ("UNEMP_GAP", INTERACTION_REGS),
    "(D) UnempGap, with controls":  ("UNEMP_GAP", INTERACTION_REGS + CONTROLS),
}

all_results = {}
for spec_label, (outcome, regs) in specs.items():
    res = run_ols(outcome, regs)
    all_results[spec_label] = (outcome, res)

    print(f"\n\n{'─'*65}")
    print(f"  Spec {spec_label}")
    print(f"  Outcome: {outcome}  |  Controls: {'Yes' if len(regs)>3 else 'No'}")
    print(f"{'─'*65}")
    b1c, b1p = print_coef(res, "NewPost",  "beta1 (New_m × Post)     [causal]")
    b2c, b2p = print_coef(res, "ContPost", "beta2 (Cont_m × Post)    [combined]")
    b3c, b3p = print_coef(res, "ExitPost", "beta3 (Exit_m × Post)    [falsif.]")
    if len(regs) > 3:
        print(f"    --- Controls ---")
        for ctrl in CONTROLS:
            if ctrl in res.params.index:
                c=res.params[ctrl]; s=res.bse[ctrl]; p=res.pvalues[ctrl]
                print(f"    {ctrl:35s}: {c:+.4f} ({s:.4f}) {stars(p)}")
    print(f"    {'Obs / R²':35s}: {int(res.nobs)} / {res.rsquared:.4f}")

    if b1c is not None and b2c is not None:
        diff = abs(b2c) - abs(b1c)
        print(f"\n    |beta2| - |beta1| = {diff:+.4f}  "
              f"({'Cont > New as expected' if diff>0 else 'New > Cont — unexpected'})")
    if b3c is not None:
        print(f"    Falsification (Exit): {'PASSES (p>=0.10)' if b3p>=0.10 else 'FAILS (p<0.10) — pre-trend concern'}")

# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE (publication style)
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 75)
print("HETEROGENEOUS DiD — SUMMARY TABLE")
print("(cluster-robust SE at municipality level in parentheses)")
print("=" * 75)

spec_keys = list(specs.keys())
hdrs     = ["(A)","(B)","(C)","(D)"]
outcomes = ["EmpGap","EmpGap","UnempGap","UnempGap"]
ctrls_h  = ["No","Yes","No","Yes"]

def fc(res, name):
    if name not in res.params.index: return "—"
    return f"{res.params[name]:+.3f}{stars(res.pvalues[name])}"
def fs(res, name):
    if name not in res.params.index: return ""
    return f"({res.bse[name]:.3f})"

row_res = [all_results[k][1] for k in spec_keys]

print(f"{'':35} " + "  ".join(f"{h:>13}" for h in hdrs))
print(f"{'Outcome':35} " + "  ".join(f"{o:>13}" for o in outcomes))
print(f"{'Controls':35} " + "  ".join(f"{c:>13}" for c in ctrls_h))
print("-" * 90)
for vname, vlabel in [
    ("NewPost",  "beta1  New_m × Post  [causal est.]"),
    ("",         ""),
    ("ContPost", "beta2  Cont_m × Post [combined]"),
    ("",         ""),
    ("ExitPost", "beta3  Exit_m × Post [falsif.]"),
    ("",         ""),
]:
    if vname == "":
        print(f"{'':35} " + "  ".join(f"{fs(r, prev_v):>13}" for r in row_res))
        continue
    prev_v = vname
    print(f"{vlabel:35} " + "  ".join(f"{fc(r, vname):>13}" for r in row_res))

print(f"{'R-squared':35} " + "  ".join(f"{r.rsquared:>13.4f}" for r in row_res))
print(f"{'Observations':35} " + "  ".join(f"{int(r.nobs):>13d}" for r in row_res))
print(f"{'Municipality FE':35} " + "  ".join(f"{'Yes':>13}" for _ in row_res))
print(f"{'Year FE':35} " + "  ".join(f"{'Yes':>13}" for _ in row_res))
print("=" * 75)
print("Notes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")
print(f"  Post_t = 1 for t >= {POST_YEAR}")
print(f"  New (n=4): Helsingør, Guldborgsund, Vejle, Silkeborg")
print(f"  Cont (n=11): Copenhagen, H.Taastrup, Holbæk, Slagelse, Odense,")
print(f"               Sønderborg, Esbjerg, Horsens, Kolding, Aarhus, Viborg")
print(f"  Exit (n=6): Greve, Fredericia, Svendborg, Thisted, Aalborg, Ishøj")
print(f"  Control (n={df[df.GROUP=='Control']['MUN_CODE'].nunique()}): all remaining municipalities")

# ══════════════════════════════════════════════════════════════
# FIGURE — Coefficient plot
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Heterogeneous Treatment Effects DiD\n"
    r"$\beta_1$ (New), $\beta_2$ (Cont), $\beta_3$ (Exit — falsification)  |  Post = $t \geq 2018$",
    fontsize=12, fontweight="bold"
)

for ax, (spec_key_no, spec_key_yes), outcome_label in [
    (axes[0],
     ("(A) EmpGap,   no controls", "(B) EmpGap,   with controls"),
     "Employment Gap (pp)"),
    (axes[1],
     ("(C) UnempGap, no controls", "(D) UnempGap, with controls"),
     "Unemployment Gap (pp)"),
]:
    groups    = ["NewPost", "ContPost", "ExitPost"]
    glabels   = [r"$\beta_1$ New", r"$\beta_2$ Cont", r"$\beta_3$ Exit (falsif.)"]
    colors_no = ["#2563EB","#7C3AED","#DC2626"]
    colors_y  = ["#60A5FA","#A78BFA","#FCA5A5"]

    x = np.arange(len(groups))
    w = 0.35

    for offset, spec_key, colors, lbl in [
        (-w/2, spec_key_no,  colors_no, "No controls"),
        (+w/2, spec_key_yes, colors_y,  "With controls"),
    ]:
        res = all_results[spec_key][1]
        coefs = [res.params.get(g, np.nan) for g in groups]
        ses   = [res.bse.get(g, np.nan)    for g in groups]
        pvs   = [res.pvalues.get(g, 1.0)   for g in groups]
        ci_lo = [coefs[i] - 1.96*ses[i] for i in range(len(groups))]
        ci_hi = [coefs[i] + 1.96*ses[i] for i in range(len(groups))]

        for i in range(len(groups)):
            c = colors[i]
            bar = ax.bar(x[i]+offset, coefs[i], w*0.85,
                         color=c, alpha=0.85, edgecolor="white", lw=0.8,
                         label=lbl if i == 0 else "")
            ax.plot([x[i]+offset, x[i]+offset], [ci_lo[i], ci_hi[i]],
                    color="black", lw=1.5, zorder=5)
            if pvs[i] < 0.05:
                ax.text(x[i]+offset, ci_hi[i]+0.05,
                        stars(pvs[i]), ha="center", fontsize=10,
                        color="black", fontweight="bold")

    ax.axhline(0, color="black", lw=1)
    ax.axvline(1.5, color="#D1D5DB", lw=1, linestyle="--", alpha=0.7)
    ax.text(1.5, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1]>0 else 0.5,
            "Falsification →", ha="center", fontsize=8, color="#6B7280")

    ax.set_title(outcome_label, fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(glabels, fontsize=10)
    ax.set_ylabel("DiD coefficient (pp, relative to control)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = BASE / "heterogeneous_treatment_plot.png"
fig.savefig(fig_path, dpi=160, bbox_inches="tight")
print(f"\nFigure saved: {fig_path}")
plt.close()

print("\nDone.")
