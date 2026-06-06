"""
Regression Analysis — Danish Municipality Panel 2010-2022
Eq (1): Baseline TWFE DiD
        Gap_mt = alpha_m + alpha_t + beta(Treated_m x Post_t) + gamma*X_mt + e_mt
Eq (2): Event Study DiD
        Gap_mt = alpha_m + alpha_t + sum_{tau!=2017} delta_tau(Treated_m x 1[t=tau]) + gamma*X_mt + e_mt
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

OUTCOMES   = {"EMP_GAP": "Employment Gap", "UNEMP_GAP": "Unemployment Gap"}
CONTROLS   = ["SH_SHARE","LOG_INCOME","NW_IMMIG_SHARE"]
POST_YEAR  = 2018   # Post = 1 for t >= 2018 (ghetto-plan intensification)
REF_YEAR   = 2017   # omitted reference year in event study
YEARS      = sorted(df["YEAR"].unique())

# ── Treatment × Post dummy ────────────────────────────────────
df["Post"]      = (df["YEAR"] >= POST_YEAR).astype(float)
df["TreatPost"] = df["TREAT"] * df["Post"]

# ── Fixed-effect dummies (drop first to avoid perfect collinearity) ──
mun_dum  = pd.get_dummies(df["MUN_CODE"], prefix="mun",  drop_first=True).astype(float)
year_dum = pd.get_dummies(df["YEAR"],     prefix="year", drop_first=True).astype(float)

print("Dataset:", df.shape[0], "obs |", df["MUN_CODE"].nunique(), "municipalities |",
      df["YEAR"].nunique(), "years")
print("Treated municipalities:", df[df.TREAT==1]["MUN_CODE"].nunique())
print("Control municipalities:", df[df.TREAT==0]["MUN_CODE"].nunique())
print(f"Post period defined as: t >= {POST_YEAR}")
print(f"Event-study reference year: {REF_YEAR}\n")

# ══════════════════════════════════════════════════════════════
# HELPER: cluster-robust OLS with municipality & year FEs
# ══════════════════════════════════════════════════════════════
def run_ols(outcome, extra_regressors, label=""):
    X = pd.concat([df[extra_regressors], mun_dum, year_dum], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[outcome]
    valid = y.notna() & X.notna().all(axis=1)

    result = sm.OLS(y[valid], X[valid]).fit(
        cov_type="cluster",
        cov_kwds={"groups": df.loc[valid, "MUN_CODE"]}
    )

    # Within R2: ratio of explained to total after demeaning
    # Use reported R2 directly (includes FE dummy variation)
    return result, valid

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return ""

# ══════════════════════════════════════════════════════════════
# EQUATION 1: Baseline TWFE DiD
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("EQUATION (1) — Baseline TWFE Difference-in-Differences")
print("Gap_mt = alpha_m + alpha_t + beta(Treated_m x Post_t) + gamma*X + e")
print("=" * 65)

specs = {
    "(1a) EmpGap, no controls":    ("EMP_GAP",   ["TreatPost"]),
    "(1b) EmpGap, with controls":  ("EMP_GAP",   ["TreatPost"] + CONTROLS),
    "(1c) UnempGap, no controls":  ("UNEMP_GAP", ["TreatPost"]),
    "(1d) UnempGap, with controls":("UNEMP_GAP", ["TreatPost"] + CONTROLS),
}

baseline_results = {}
for spec_label, (outcome, regressors) in specs.items():
    res, valid = run_ols(outcome, regressors)
    coef = res.params["TreatPost"]
    se   = res.bse["TreatPost"]
    tval = res.tvalues["TreatPost"]
    pval = res.pvalues["TreatPost"]
    ci   = res.conf_int().loc["TreatPost"]
    baseline_results[spec_label] = {
        "outcome": outcome, "coef": coef, "se": se, "t": tval,
        "p": pval, "ci_lo": ci[0], "ci_hi": ci[1],
        "r2": res.rsquared, "n": int(res.nobs)
    }

    print(f"\n  Spec {spec_label}")
    print(f"  Outcome          : {outcome}")
    print(f"  Controls         : {'Yes' if len(regressors) > 1 else 'No'}")
    print(f"  beta (Treat×Post): {coef:+.4f}  SE = {se:.4f}  t = {tval:.3f}  p = {pval:.4f} {stars(pval)}")
    print(f"  95% CI           : [{ci[0]:.4f},  {ci[1]:.4f}]")
    print(f"  Obs              : {int(res.nobs)}   R2 = {res.rsquared:.4f}")

    if len(regressors) > 1:
        print(f"  --- Control coefficients ---")
        for ctrl in CONTROLS:
            if ctrl in res.params.index:
                c2, s2, p2 = res.params[ctrl], res.bse[ctrl], res.pvalues[ctrl]
                print(f"    {ctrl:20s}: {c2:+.4f}  SE={s2:.4f}  {stars(p2)}")

# ══════════════════════════════════════════════════════════════
# EQUATION 2: Event Study DiD
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("EQUATION (2) — Event Study DiD")
print("Gap_mt = alpha_m + alpha_t + sum_{tau!=2017} delta_tau(Treated_m x 1[t=tau]) + gamma*X + e")
print("=" * 65)

# Build event-study interaction dummies
es_cols = []
for yr in YEARS:
    if yr == REF_YEAR:
        continue
    col = f"es_{yr}"
    df[col] = df["TREAT"] * (df["YEAR"] == yr).astype(float)
    es_cols.append(col)

def run_event_study(outcome, controls, spec_label):
    regressors = es_cols + controls
    res, valid = run_ols(outcome, regressors)

    deltas = {}
    for col in es_cols:
        yr = int(col.replace("es_",""))
        if col in res.params.index:
            ci = res.conf_int().loc[col]
            deltas[yr] = {
                "coef": res.params[col], "se": res.bse[col],
                "t": res.tvalues[col],   "p": res.pvalues[col],
                "ci_lo": ci[0], "ci_hi": ci[1]
            }
    deltas[REF_YEAR] = {"coef":0,"se":0,"t":0,"p":1.0,"ci_lo":0,"ci_hi":0}

    print(f"\n  Spec {spec_label}  |  Outcome: {outcome}  |  Controls: {'Yes' if controls else 'No'}")
    print(f"  {'Year':>6}  {'delta_tau':>10}  {'SE':>8}  {'t':>7}  {'95% CI':^20}  {'p':>7}  Sig")
    print(f"  {'-'*70}")
    for yr in sorted(deltas.keys()):
        d = deltas[yr]
        ref = " <- REF" if yr == REF_YEAR else ""
        ci_str = f"[{d['ci_lo']:+6.3f}, {d['ci_hi']:+6.3f}]"
        print(f"  {yr:>6}  {d['coef']:>+10.4f}  {d['se']:>8.4f}  {d['t']:>7.3f}  "
              f"{ci_str:^20}  {d['p']:>7.4f}  {stars(d['p'])}{ref}")
    print(f"  Obs: {int(res.nobs)}   R2: {res.rsquared:.4f}")

    # Pre-trend test (years before POST_YEAR, excluding REF_YEAR)
    pre_years = [y for y in YEARS if y < POST_YEAR and y != REF_YEAR]
    pre_coefs = [deltas[y]["coef"] for y in pre_years if y in deltas]
    pre_pvals = [deltas[y]["p"]    for y in pre_years if y in deltas]
    print(f"\n  PRE-TREND CHECK (years {min(pre_years)}-{max(pre_years)}, tau != {REF_YEAR}):")
    for yr in pre_years:
        if yr in deltas:
            d = deltas[yr]
            print(f"    {yr}: delta = {d['coef']:+.4f}  p = {d['p']:.4f}  {stars(d['p'])}")
    joint_pre = np.mean(np.abs(pre_coefs))
    sig_pre   = sum(p < 0.05 for p in pre_pvals)
    print(f"  Mean |delta| pre-period: {joint_pre:.4f}  |  Significant pre-trends: {sig_pre}/{len(pre_pvals)}")

    return deltas, res

es1_deltas, es1_res = run_event_study("EMP_GAP",   [],       "(2a)")
es2_deltas, es2_res = run_event_study("EMP_GAP",   CONTROLS, "(2b)")
es3_deltas, es3_res = run_event_study("UNEMP_GAP", CONTROLS, "(2c)")

# ══════════════════════════════════════════════════════════════
# FIGURE: Event Study Plot (2×2 grid)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle(
    "Event Study DiD — Employment Gap (regression reference year 2017)\n"
    "Treated=15,Control=83 |  95% CI with cluster-robust SE",
    fontsize=13, fontweight="bold"
)

for ax, deltas, title in [
    (axes[0], es1_deltas, "Eq. (2a): Without controls"),
    (axes[1], es2_deltas, "Eq. (2b): With controls"),
]:
    yrs   = sorted(deltas.keys())
    coefs = [deltas[y]["coef"]  for y in yrs]
    ci_lo = [deltas[y]["ci_lo"] for y in yrs]
    ci_hi = [deltas[y]["ci_hi"] for y in yrs]
    pvs   = [deltas[y]["p"]     for y in yrs]

    # Shade pre vs post
    ax.axvspan(min(yrs)-0.5, POST_YEAR-0.5, alpha=0.06, color="#DC2626", label="Pre-period")
    ax.axvspan(POST_YEAR-0.5, max(yrs)+0.5, alpha=0.06, color="#2563EB", label="Post-period")
    ax.axhline(0,          color="black",  lw=1.2, linestyle="-")
    ax.axvline(REF_YEAR,   color="#6B7280",lw=1.5, linestyle="--", alpha=0.9, label=f"Ref year ({REF_YEAR})")
    ax.axvline(POST_YEAR-0.5, color="#DC2626", lw=1.2, linestyle=":", alpha=0.7)

    for i, yr in enumerate(yrs):
        c = "#6B7280" if yr == REF_YEAR else ("#DC2626" if yr < POST_YEAR else "#2563EB")
        ax.plot([yr, yr], [ci_lo[i], ci_hi[i]], color=c, lw=2, alpha=0.8, zorder=3)
        mk = "D" if yr == REF_YEAR else "o"
        ax.plot(yr, coefs[i], marker=mk, color=c, ms=8 if yr == REF_YEAR else 7,
                zorder=5, markeredgecolor="white", markeredgewidth=0.8)
        if pvs[i] < 0.05 and yr != REF_YEAR:
            ax.text(yr, ci_hi[i]+0.15, "*", ha="center", fontsize=11, color=c, fontweight="bold")

    ax.plot(yrs, coefs, color="#9CA3AF", lw=1, linestyle="-", zorder=1, alpha=0.4)

    # Annotation: pre-trend label
    ax.text(2012.5, ax.get_ylim()[0]+0.1 if ax.get_ylim()[0]>-20 else -7,
            "Pre-treatment\neffects", ha="center", fontsize=8, color="#DC2626", alpha=0.7)
    ax.text(2019, ax.get_ylim()[0]+0.1 if ax.get_ylim()[0]>-20 else -7,
            "Post-treatment\neffects", ha="center", fontsize=8, color="#2563EB", alpha=0.7)

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(r"$\delta_{\tau}$ (pp, relative to 2017)", fontsize=11)
    ax.set_xticks(yrs)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.01, 0.99),
              borderaxespad=0.3, framealpha=0.85)

plt.tight_layout()
fig.savefig(BASE / "event_study_plot.png", dpi=160, bbox_inches="tight")
print(f"\n\nEvent study figure: {BASE / 'event_study_plot.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE (publication-style)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("REGRESSION SUMMARY TABLE  (cluster-robust SE in parentheses)")
print("=" * 72)

specs_ordered = list(specs.keys())
headers = ["(1a)", "(1b)", "(1c)", "(1d)"]
outcomes_h = ["EmpGap", "EmpGap", "UnempGap", "UnempGap"]
controls_h = ["No", "Yes", "No", "Yes"]

print(f"{'':30} " + "  ".join(f"{h:>14}" for h in headers))
print(f"{'Outcome':30} " + "  ".join(f"{o:>14}" for o in outcomes_h))
print(f"{'Controls':30} " + "  ".join(f"{c:>14}" for c in controls_h))
print("-" * 90)

def fmt_coef(coef, se, p):
    s = stars(p)
    return f"{coef:+.3f}{s}"
def fmt_se(se):
    return f"({se:.3f})"

# beta row
row_beta = [baseline_results[k] for k in specs_ordered]
print(f"{'Treated x Post (beta)':30} " +
      "  ".join(f"{fmt_coef(r['coef'],r['se'],r['p']):>14}" for r in row_beta))
print(f"{'':30} " +
      "  ".join(f"{fmt_se(r['se']):>14}" for r in row_beta))
# R2 row
print(f"{'R-squared':30} " +
      "  ".join(f"{r['r2']:>14.4f}" for r in row_beta))
# Obs row
print(f"{'Observations':30} " +
      "  ".join(f"{r['n']:>14d}" for r in row_beta))
# FE rows
print(f"{'Municipality FE':30} " + "  ".join(f"{'Yes':>14}" for _ in row_beta))
print(f"{'Year FE':30} " + "  ".join(f"{'Yes':>14}" for _ in row_beta))
print("=" * 72)
print(f"Notes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")
print(f"  Post_t = 1 for t >= {POST_YEAR} (ghetto-plan intensification)")
print(f"  Controls: Social housing share, Log income, NW immigrant share")
print(f"  SE clustered at municipality level (98 clusters)")

print("\nDone.")
