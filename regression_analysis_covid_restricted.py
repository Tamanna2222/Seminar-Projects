"""
Regression Analysis — COVID-CONTROLLED VERSION
Adds Treated_m × COVID_t to both equations to absorb any differential
COVID shock, so that beta (Eq 1) and delta_tau (Eq 2) capture
only the legislation effect.

COVID_t = 1 for t in {2020, 2021}

Eq (1):  Gap_mt = alpha_m + alpha_t
                + beta  (Treated_m x Post_t)
                + phi   (Treated_m x COVID_t)     <-- NEW
                + gamma X_mt + e_mt

Eq (2):  Gap_mt = alpha_m + alpha_t
                + sum_{tau not in {2017,2020,2021}} delta_tau (Treated_m x 1[t=tau])
                + phi   (Treated_m x COVID_t)     <-- replaces es_2020, es_2021
                + gamma X_mt + e_mt
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
df_full = pd.read_csv(BASE / "panel_mun_year_with_ras206.csv")
df_full = df_full.sort_values(["MUN_CODE","YEAR"]).reset_index(drop=True)


# ── Restriction: keep treated + controls with SH>10% AND NW>5% at 2010 ──
SH_THRESHOLD = 5
NW_THRESHOLD  = 5
_b2010 = (
    df_full[df_full.YEAR == 2010]
    [["MUN_CODE","TREAT","SH_SHARE","NW_IMMIG_SHARE"]]
    .copy()
    .rename(columns={"SH_SHARE":"sh_2010","NW_IMMIG_SHARE":"nw_2010"})
)
_ctrl = _b2010[_b2010.TREAT == 0].copy()
_ctrl["flag_drop"] = (_ctrl["sh_2010"] <= SH_THRESHOLD) | (_ctrl["nw_2010"] <= NW_THRESHOLD)
_keep = set(
    _b2010[_b2010.TREAT == 1]["MUN_CODE"].tolist() +
    _ctrl[~_ctrl["flag_drop"]]["MUN_CODE"].tolist()
)
df = df_full[df_full["MUN_CODE"].isin(_keep)].copy().reset_index(drop=True)
_n_treat = df[df.TREAT==1]['MUN_CODE'].nunique()
_n_ctrl  = df[df.TREAT==0]['MUN_CODE'].nunique()
print(f"  RESTRICTED SAMPLE: Treated={_n_treat}, Control={_n_ctrl}, Obs={len(df)}")
print(f"  Restriction: SH_SHARE_2010 > {SH_THRESHOLD}% AND NW_IMMIG_SHARE_2010 > {NW_THRESHOLD}%")

CONTROLS   = ["SH_SHARE","LOG_INCOME","NW_IMMIG_SHARE"]
POST_YEAR  = 2018
REF_YEAR   = 2017
COVID_YEARS = {2020, 2021}           # years absorbed by COVID dummy
YEARS       = sorted(df["YEAR"].unique())

# ── Core dummies ──────────────────────────────────────────────
df["Post"]       = (df["YEAR"] >= POST_YEAR).astype(float)
df["COVID"]      = df["YEAR"].isin(COVID_YEARS).astype(float)
df["TreatPost"]  = df["TREAT"] * df["Post"]
df["TreatCOVID"] = df["TREAT"] * df["COVID"]   # absorbs differential COVID shock

mun_dum  = pd.get_dummies(df["MUN_CODE"], prefix="mun",  drop_first=True).astype(float)
year_dum = pd.get_dummies(df["YEAR"],     prefix="year", drop_first=True).astype(float)

print("=" * 68)
print("COVID-CONTROLLED REGRESSION — Danish Municipality Panel 2010-2022")
print("=" * 68)
print(f"  Post period:   t >= {POST_YEAR}")
print(f"  Ref year:      {REF_YEAR}  (omitted from event study)")
print(f"  COVID years:   {sorted(COVID_YEARS)}  (absorbed by Treated x COVID dummy)")
print(f"  N obs:         {len(df)}  |  Municipalities: {df['MUN_CODE'].nunique()}")
print(f"  Treated:       {df[df.TREAT==1]['MUN_CODE'].nunique()}")
print(f"  Control:       {df[df.TREAT==0]['MUN_CODE'].nunique()}")

# ── Helpers ───────────────────────────────────────────────────
def stars(p):
    if p<0.001: return "***"
    if p<0.01:  return "**"
    if p<0.05:  return "*"
    if p<0.10:  return "."
    return ""

def run_ols(outcome, extra_regressors):
    X = pd.concat([df[extra_regressors], mun_dum, year_dum], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[outcome]
    valid = y.notna() & X.notna().all(axis=1)
    res = sm.OLS(y[valid], X[valid]).fit(
        cov_type="cluster",
        cov_kwds={"groups": df.loc[valid,"MUN_CODE"]}
    )
    return res, valid

def print_param(res, name, label=None):
    label = label or name
    if name not in res.params.index:
        print(f"    {label:30s}: [not in model]")
        return
    c,s,t,p = res.params[name], res.bse[name], res.tvalues[name], res.pvalues[name]
    ci = res.conf_int().loc[name]
    print(f"    {label:30s}: {c:+.4f}  SE={s:.4f}  t={t:.3f}  p={p:.4f} {stars(p)}")
    print(f"    {'':30s}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

# ══════════════════════════════════════════════════════════════
# EQUATION (1) — Baseline TWFE DiD  +  Treated × COVID
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 68)
print("EQUATION (1) — Baseline TWFE DiD with COVID control")
print("Gap_mt = a_m + a_t + beta(Treat×Post) + phi(Treat×COVID) + gamma*X + e")
print("=" * 68)

specs_eq1 = {
    "(1a) EmpGap,   no controls":    ("EMP_GAP",   ["TreatPost","TreatCOVID"]),
    "(1b) EmpGap,   with controls":  ("EMP_GAP",   ["TreatPost","TreatCOVID"] + CONTROLS),
    "(1c) UnempGap, no controls":    ("UNEMP_GAP", ["TreatPost","TreatCOVID"]),
    "(1d) UnempGap, with controls":  ("UNEMP_GAP", ["TreatPost","TreatCOVID"] + CONTROLS),
}

eq1_results = {}
for label,(outcome,regs) in specs_eq1.items():
    res, _ = run_ols(outcome, regs)
    eq1_results[label] = res
    print(f"\n  [{label}]  Outcome={outcome}")
    print_param(res, "TreatPost",  "beta  (Treated x Post)")
    print_param(res, "TreatCOVID","phi   (Treated x COVID)")
    if len(regs) > 2:
        for c in CONTROLS:
            if c in res.params.index:
                v = res.params[c]; s = res.bse[c]; p = res.pvalues[c]
                print(f"    {c:30s}: {v:+.4f} ({s:.4f}) {stars(p)}")
    print(f"    {'Obs / R²':30s}: {int(res.nobs)} / {res.rsquared:.4f}")

# ══════════════════════════════════════════════════════════════
# EQUATION (2) — Event Study  +  Treated × COVID
# Note: es_2020 and es_2021 are REPLACED by TreatCOVID to
#       avoid perfect collinearity (TreatCOVID = es_2020 + es_2021)
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 68)
print("EQUATION (2) — Event Study DiD with COVID control")
print("Gap_mt = a_m + a_t")
print("       + sum_{tau not in {REF,COVID}} delta_tau(Treat×1[t=tau])")
print("       + phi(Treat×COVID) + gamma*X + e")
print(f"  Omitted from year interactions: REF={REF_YEAR}, COVID={sorted(COVID_YEARS)}")
print("=" * 68)

# Build event-study cols — exclude REF_YEAR and COVID_YEARS
es_cols = []
for yr in YEARS:
    if yr == REF_YEAR or yr in COVID_YEARS:
        continue
    col = f"es_{yr}"
    df[col] = df["TREAT"] * (df["YEAR"] == yr).astype(float)
    es_cols.append(col)

def run_event_study_covid(outcome, controls, spec_label):
    regs = es_cols + ["TreatCOVID"] + controls
    res, _ = run_ols(outcome, regs)

    # Collect delta_tau — non-COVID years
    deltas = {}
    for col in es_cols:
        yr = int(col.replace("es_",""))
        if col in res.params.index:
            ci = res.conf_int().loc[col]
            deltas[yr] = {"coef": res.params[col], "se": res.bse[col],
                          "t": res.tvalues[col], "p": res.pvalues[col],
                          "ci_lo": ci[0], "ci_hi": ci[1], "type": "es"}

    # Add reference year (zero by construction)
    deltas[REF_YEAR] = {"coef":0,"se":0,"t":0,"p":1.0,
                         "ci_lo":0,"ci_hi":0,"type":"ref"}

    # Add COVID as a separate entry for plotting
    for yr in sorted(COVID_YEARS):
        if "TreatCOVID" in res.params.index:
            ci = res.conf_int().loc["TreatCOVID"]
            deltas[yr] = {"coef": res.params["TreatCOVID"],
                          "se":   res.bse["TreatCOVID"],
                          "t":    res.tvalues["TreatCOVID"],
                          "p":    res.pvalues["TreatCOVID"],
                          "ci_lo": ci[0], "ci_hi": ci[1],
                          "type": "covid"}

    # Print table
    print(f"\n  Spec {spec_label}  |  Outcome: {outcome}  |  Controls: {'Yes' if controls else 'No'}")
    print(f"  {'Year':>6}  {'delta / phi':>11}  {'SE':>8}  {'t':>7}  {'95% CI':^22}  p       Sig  Note")
    print(f"  {'-'*80}")
    for yr in sorted(deltas.keys()):
        d = deltas[yr]
        ci_s = f"[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]"
        note = ""
        if d["type"] == "ref":   note = "<- REF"
        elif d["type"] == "covid": note = "<- phi (COVID avg)"
        print(f"  {yr:>6}  {d['coef']:>+11.4f}  {d['se']:>8.4f}  {d['t']:>7.3f}  "
              f"{ci_s:^22}  {d['p']:.4f}  {stars(d['p']):3s}  {note}")

    # Phi (COVID interaction)
    if "TreatCOVID" in res.params.index:
        phi  = res.params["TreatCOVID"]
        sphi = res.bse["TreatCOVID"]
        pphi = res.pvalues["TreatCOVID"]
        print(f"\n  phi (Treated x COVID 2020-21): {phi:+.4f}  SE={sphi:.4f}  p={pphi:.4f} {stars(pphi)}")

    # Pre-trend check (non-COVID pre-period)
    pre_yrs = [y for y in YEARS if y < POST_YEAR and y != REF_YEAR and y not in COVID_YEARS]
    pre_d   = [deltas[y]["coef"] for y in pre_yrs if y in deltas]
    pre_p   = [deltas[y]["p"]    for y in pre_yrs if y in deltas]
    sig_pre = sum(p < 0.05 for p in pre_p)
    print(f"  PRE-TREND (non-COVID): mean |delta| = {np.mean(np.abs(pre_d)):.4f}  "
          f"| Significant: {sig_pre}/{len(pre_p)}")
    print(f"  Obs: {int(res.nobs)}  R2: {res.rsquared:.4f}")

    return deltas, res

es1_d, es1_r = run_event_study_covid("EMP_GAP",   [],       "(2a)")
es2_d, es2_r = run_event_study_covid("EMP_GAP",   CONTROLS, "(2b)")
es3_d, es3_r = run_event_study_covid("UNEMP_GAP", CONTROLS, "(2c)")

# ══════════════════════════════════════════════════════════════
# FIGURE — Event Study Plot (with COVID band)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle(
    r"Event Study DiD — COVID-DUMMY(restricted control sample)"
    "\nTreated=15, Control=24  |  95% CI with cluster-robust SE",
    fontsize=12, fontweight="bold"
)

for ax, deltas, title in [
    (axes[0], es1_d, "Eq. (2a): Without controls"),
    (axes[1], es2_d, "Eq. (2b): With controls"),
]:
    yrs   = sorted(deltas.keys())
    coefs = [deltas[y]["coef"]  for y in yrs]
    ci_lo = [deltas[y]["ci_lo"] for y in yrs]
    ci_hi = [deltas[y]["ci_hi"] for y in yrs]
    pvs   = [deltas[y]["p"]     for y in yrs]
    types = [deltas[y]["type"]  for y in yrs]

    # Background shading
    ax.axvspan(min(yrs)-0.5, POST_YEAR-0.5, alpha=0.06, color="#DC2626", label="Pre-period")
    ax.axvspan(POST_YEAR-0.5, max(yrs)+0.5, alpha=0.06, color="#2563EB", label="Post-period")
    # COVID band
    ax.axvspan(min(COVID_YEARS)-0.45, max(COVID_YEARS)+0.45,
               alpha=0.18, color="#F59E0B", zorder=0, label="COVID (φ)")
    ax.axhline(0,        color="black",   lw=1.2, linestyle="-")
    ax.axvline(REF_YEAR, color="#6B7280", lw=1.5, linestyle="--",
               alpha=0.9, label=f"Ref year ({REF_YEAR})")
    ax.axvline(POST_YEAR-0.5, color="#DC2626", lw=1.2, linestyle=":", alpha=0.6)

    for i, yr in enumerate(yrs):
        t = types[i]
        if t == "ref":     c = "#6B7280"; mk = "D"; ms = 8
        elif t == "covid": c = "#D97706"; mk = "s"; ms = 8   # amber square for COVID
        elif yr < POST_YEAR: c = "#DC2626"; mk = "o"; ms = 7
        else:              c = "#2563EB"; mk = "o"; ms = 7

        ax.plot([yr, yr], [ci_lo[i], ci_hi[i]], color=c, lw=2, alpha=0.8, zorder=3)
        ax.plot(yr, coefs[i], marker=mk, color=c, ms=ms,
                zorder=5, markeredgecolor="white", markeredgewidth=0.8)
        if pvs[i] < 0.05 and t not in ("ref",):
            ax.text(yr, ci_hi[i]+0.15, "*", ha="center", fontsize=11,
                    color=c, fontweight="bold")

    # Connect non-COVID dots with a line
    non_covid_yrs  = [y for y in yrs if deltas[y]["type"] != "covid"]
    non_covid_coef = [deltas[y]["coef"] for y in non_covid_yrs]
    ax.plot(non_covid_yrs, non_covid_coef, color="#9CA3AF",
            lw=1, linestyle="-", zorder=1, alpha=0.4)

    # Annotations
    ymin = ax.get_ylim()[0]
    ax.text(2012.5, ymin + 0.3,
            "Pre-treatment\neffects", ha="center", fontsize=8,
            color="#DC2626", alpha=0.75)
    ax.text(2020.5, ymin + 0.3,
            "φ\n(COVID)", ha="center", fontsize=8, color="#D97706", alpha=0.9)
    ax.text(2019,   ymin + 0.3,
            "Post-treatment\neffects", ha="center", fontsize=7.5,
            color="#2563EB", alpha=0.75)

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(r"$\delta_{\tau}$ / $\phi$ (pp, relative to 2017)", fontsize=11)
    ax.set_xticks(yrs)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, loc="upper left",
              bbox_to_anchor=(0.01, 0.99), borderaxespad=0.3, framealpha=0.85)

plt.tight_layout()
fig_path = BASE / "event_study_plot_covid_restricted.png"
fig.savefig(fig_path, dpi=160, bbox_inches="tight")
print(f"\n\nFigure saved: {fig_path}")
plt.close()

# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("REGRESSION SUMMARY TABLE — COVID-CONTROLLED + RESTRICTED SAMPLE")
print("=" * 72)
specs_ordered = list(specs_eq1.keys())
headers  = ["(1a)","(1b)","(1c)","(1d)"]
outcomes = ["EmpGap","EmpGap","UnempGap","UnempGap"]
ctrls_h  = ["No","Yes","No","Yes"]

def fmt_c(res, name):
    if name not in res.params.index: return "—"
    c = res.params[name]; s = res.bse[name]; p = res.pvalues[name]
    return f"{c:+.3f}{stars(p)}"
def fmt_s(res, name):
    if name not in res.params.index: return ""
    return f"({res.bse[name]:.3f})"

row_res = [eq1_results[k] for k in specs_ordered]

print(f"{'':30} " + "  ".join(f"{h:>15}" for h in headers))
print(f"{'Outcome':30} " + "  ".join(f"{o:>15}" for o in outcomes))
print(f"{'Controls':30} " + "  ".join(f"{c:>15}" for c in ctrls_h))
print("-" * 95)
print(f"{'beta (Treated x Post)':30} " +
      "  ".join(f"{fmt_c(r,'TreatPost'):>15}" for r in row_res))
print(f"{'':30} " +
      "  ".join(f"{fmt_s(r,'TreatPost'):>15}" for r in row_res))
print(f"{'phi  (Treated x COVID)':30} " +
      "  ".join(f"{fmt_c(r,'TreatCOVID'):>15}" for r in row_res))
print(f"{'':30} " +
      "  ".join(f"{fmt_s(r,'TreatCOVID'):>15}" for r in row_res))
print(f"{'R-squared':30} " +
      "  ".join(f"{r.rsquared:>15.4f}" for r in row_res))
print(f"{'Observations':30} " +
      "  ".join(f"{int(r.nobs):>15d}" for r in row_res))
print(f"{'Municipality FE':30} " + "  ".join(f"{'Yes':>15}" for _ in row_res))
print(f"{'Year FE':30} " + "  ".join(f"{'Yes':>15}" for _ in row_res))
print(f"{'COVID control (Treat×COVID)':30} " + "  ".join(f"{'Yes':>15}" for _ in row_res))
print("=" * 72)
print("Notes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")
print(f"  Post_t  = 1 for t >= {POST_YEAR};  COVID_t = 1 for t in {sorted(COVID_YEARS)}")
print(f"  SE clustered at municipality level (98 clusters)")
print(f"  In Eq(2): es_2020 and es_2021 replaced by single Treated x COVID dummy")

print("\nDone.")
