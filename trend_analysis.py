"""
Trend Analysis: Employment Gap, Syrian Refugee Crisis (2015-16),
COVID-19 (2020), and Long-run Narrowing — Danish Municipality Panel
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
BASE = Path(r"d:/ONGOING COURSES/Economic Policy and the Welfare State/output")
df   = pd.read_csv(BASE / "panel_mun_year_with_ras206.csv")

# ── Colour palette ────────────────────────────────────────────
C_NAT  = "#2563EB"   # native   – blue
C_NW   = "#DC2626"   # NW imm.  – red
C_GAP  = "#7C3AED"   # gap      – violet
C_TREA = "#D97706"   # treated  – amber
C_CTRL = "#059669"   # control  – green
C_GREY = "#9CA3AF"
SHADE  = "#FEF3C7"   # event shade

EVENTS = {
    "Syrian\nRefugee\nCrisis": (2015, 2016),
    "COVID-19\nShock":         (2020, 2021),
}

def shade_events(ax, ymin=None, ymax=None):
    for label, (y0, y1) in EVENTS.items():
        ax.axvspan(y0, y1 +1, alpha=0.18, color="#FCD34D", zorder=0)
        ax.text((y0 + y1) / 2, ax.get_ylim()[1] * 0.97,
                label, ha="center", va="top", fontsize=7.5,
                color="#92400E", fontweight="bold")

YEARS = sorted(df["YEAR"].unique())

# ══════════════════════════════════════════════════════════════
# 1. Annual means by group
# ══════════════════════════════════════════════════════════════
annual = (df.groupby("YEAR")
            .agg(emp_gap        = ("EMP_GAP",        "mean"),
                 unemp_gap      = ("UNEMP_GAP",      "mean"),
                 native_emp     = ("NATIVE_EMP_RATE", "mean"),
                 nw_emp         = ("NW_EMP_RATE",     "mean"),
                 native_unemp   = ("NATIVE_UNEMP",    "mean"),
                 nw_unemp       = ("NW_UNEMP",        "mean"),
                 sh_share       = ("SH_SHARE",        "mean"),
                 nw_share       = ("NW_IMMIG_SHARE",  "mean"))
            .reset_index())

by_treat = (df.groupby(["YEAR","TREAT"])
              .agg(emp_gap    = ("EMP_GAP",        "mean"),
                   unemp_gap  = ("UNEMP_GAP",      "mean"),
                   native_emp = ("NATIVE_EMP_RATE", "mean"),
                   nw_emp     = ("NW_EMP_RATE",     "mean"))
              .reset_index())

treated = by_treat[by_treat.TREAT == 1]
control = by_treat[by_treat.TREAT == 0]

# ── Print numerical table ─────────────────────────────────────
print("="*80)
print("TREND TABLE: Annual Means (All Municipalities)")
print("="*80)
print(annual.round(2).to_string(index=False))

print("\n" + "="*80)
print("TREND TABLE: Employment Gap by Group")
print("="*80)
for yr in YEARS:
    t = treated[treated.YEAR==yr]["emp_gap"].values[0]
    c = control[control.YEAR==yr]["emp_gap"].values[0]
    a = annual[annual.YEAR==yr]["emp_gap"].values[0]
    print(f"  {yr}  All={a:.2f}pp  Treated={t:.2f}pp  Control={c:.2f}pp  Diff={t-c:.2f}pp")

# ══════════════════════════════════════════════════════════════
# 2. Year-on-year changes
# ══════════════════════════════════════════════════════════════
annual["d_emp_gap"]    = annual["emp_gap"].diff()
annual["d_nw_emp"]     = annual["nw_emp"].diff()
annual["d_native_emp"] = annual["native_emp"].diff()

print("\n" + "="*80)
print("YEAR-ON-YEAR CHANGES IN EMPLOYMENT GAP")
print("="*80)
for _, row in annual.dropna(subset=["d_emp_gap"]).iterrows():
    direction = "WIDENS" if row.d_emp_gap > 0 else "NARROWS"
    print(f"  {int(row.YEAR-1)}->{int(row.YEAR)}: Gap {direction} by {abs(row.d_emp_gap):.2f}pp"
          f"  | dNative={row.d_native_emp:+.2f}pp  dNW={row.d_nw_emp:+.2f}pp")

# ══════════════════════════════════════════════════════════════
# 3. Quantify the three episodes
# ══════════════════════════════════════════════════════════════
def val(col, yr): return annual.loc[annual.YEAR==yr, col].values[0]

# Syrian crisis: pre=2014, peak=2015, recovery=2018
pre_gap   = val("emp_gap",    2014)
peak_gap  = val("emp_gap",    2015)
rec_gap   = val("emp_gap",    2018)
pre_nw    = val("nw_emp",     2014)
peak_nw   = val("nw_emp",     2015)

# COVID: 2019 vs 2020 vs 2021
pre_covid  = val("emp_gap",   2019)
covid_gap  = val("emp_gap",   2020)
post_covid = val("emp_gap",   2021)
nw_2019    = val("nw_emp",    2019)
nw_2020    = val("nw_emp",    2020)
nat_2019   = val("native_emp",2019)
nat_2020   = val("native_emp",2020)

# Long-run narrowing: 2010 to 2022
gap_2010 = val("emp_gap",  2010)
gap_2022 = val("emp_gap",  2022)
nw_2010  = val("nw_emp",   2010)
nw_2022  = val("nw_emp",   2022)
nat_2010 = val("native_emp",2010)
nat_2022 = val("native_emp",2022)

print("\n" + "="*80)
print("QUANTIFIED EPISODE ANALYSIS")
print("="*80)
print(f"\n[SYRIAN REFUGEE CRISIS 2014-2016]")
print(f"  Employment gap pre (2014):  {pre_gap:.2f} pp")
print(f"  Employment gap peak (2015): {peak_gap:.2f} pp  (+{peak_gap-pre_gap:.2f} pp)")
print(f"  NW emp. rate 2014:          {pre_nw:.2f}%")
print(f"  NW emp. rate 2015:          {peak_nw:.2f}%  ({peak_nw-pre_nw:+.2f} pp)")
print(f"  Recovery by 2018:           {rec_gap:.2f} pp  ({rec_gap-peak_gap:+.2f} pp from peak)")

print(f"\n[COVID-19 SHOCK 2020]")
print(f"  Employment gap 2019: {pre_covid:.2f} pp")
print(f"  Employment gap 2020: {covid_gap:.2f} pp  ({covid_gap-pre_covid:+.2f} pp)")
print(f"  Employment gap 2021: {post_covid:.2f} pp  ({post_covid-pre_covid:+.2f} pp vs 2019)")
print(f"  Native emp change 2019->2020: {nat_2020-nat_2019:+.2f} pp")
print(f"  NW emp change 2019->2020:     {nw_2020-nw_2019:+.2f} pp")
print(f"  Differential impact:          {(nw_2020-nw_2019)-(nat_2020-nat_2019):+.2f} pp")

print(f"\n[LONG-RUN NARROWING 2010-2022]")
print(f"  Employment gap 2010: {gap_2010:.2f} pp")
print(f"  Employment gap 2022: {gap_2022:.2f} pp")
print(f"  Total narrowing:     {gap_2010-gap_2022:.2f} pp over 12 years")
print(f"  Native emp change:   {nat_2022-nat_2010:+.2f} pp  ({nat_2010:.2f}% -> {nat_2022:.2f}%)")
print(f"  NW emp change:       {nw_2022-nw_2010:+.2f} pp  ({nw_2010:.2f}% -> {nw_2022:.2f}%)")
print(f"  NW gain / Native gain ratio: {(nw_2022-nw_2010)/(nat_2022-nat_2010):.2f}x faster")

# ══════════════════════════════════════════════════════════════
# 4. FIGURES
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Trend Analysis: Native vs. NW Immigrant \n"
             "year 2010–2022  (n=98 municipalities)", 
             fontsize=13, fontweight="bold", y=1.01)

# ── Panel A: Employment rates & gap ──────────────────────────
ax = axes[0]
ax.plot(YEARS, annual["native_emp"], color=C_NAT, lw=2.2, marker="o", ms=4, label="Native emp. rate")
ax.plot(YEARS, annual["nw_emp"],     color=C_NW,  lw=2.2, marker="s", ms=4, label="NW immigrant emp. rate")
ax.fill_between(YEARS, annual["nw_emp"], annual["native_emp"], alpha=0.12, color=C_GAP, label="Employment gap")
for label, (y0, y1) in EVENTS.items():
    ax.axvspan(y0, y1+1, alpha=0.18, color="#FCD34D", zorder=0)
ax.set_title("A. Employment Rates & Gap (All Municipalities)", fontweight="bold")
ax.set_ylabel("Employment rate (%)")
ax.set_xlabel("Year")
ax.set_ylim(40, 90)
ax.legend(fontsize=8.5, loc="upper left")
ax.set_xticks(YEARS)
ax.tick_params(axis="x", rotation=45)
# annotate events
ax.annotate("Syrian\nRefugee\nCrisis", xy=(2015, 48), fontsize=8, color="#92400E",
            ha="center", fontweight="bold")
ax.annotate("COVID-19", xy=(2020, 45), fontsize=8, color="#92400E",
            ha="center", fontweight="bold")

# ── Panel B: Employment gap (all / treated / control) ────────
ax = axes[1]
ax.plot(YEARS, annual["emp_gap"],      color=C_GAP,  lw=2.5, marker="o", ms=4, label="All municipalities")
ax.plot(treated["YEAR"], treated["emp_gap"], color=C_TREA, lw=2,   marker="^", ms=5,
        linestyle="--", label="Treated (15 muns)")
ax.plot(control["YEAR"], control["emp_gap"], color=C_CTRL, lw=2,   marker="v", ms=5,
        linestyle="--", label="Control (83 muns)")
for label, (y0, y1) in EVENTS.items():
    ax.axvspan(y0, y1+1, alpha=0.18, color="#FCD34D", zorder=0)
ax.axhline(0, color="black", lw=0.8, linestyle=":")
ax.set_title("B. Employment Gap: Treated vs. Control Groups", fontweight="bold")
ax.set_ylabel("Gap (native − NW, pp)")
ax.set_xlabel("Year")
ax.legend(fontsize=8.5)
ax.set_xticks(YEARS)
ax.tick_params(axis="x", rotation=45)

# ── Panel C: Year-on-year gap change ─────────────────────────
ax = axes[2]
yoy = annual.dropna(subset=["d_emp_gap"])
colors_bar = [C_NW if v > 0 else C_CTRL for v in yoy["d_emp_gap"]]
bars = ax.bar(yoy["YEAR"], yoy["d_emp_gap"], color=colors_bar, alpha=0.85, edgecolor="white", lw=0.5)
ax.axhline(0, color="black", lw=1)
for label, (y0, y1) in EVENTS.items():
    ax.axvspan(y0, y1+1, alpha=0.18, color="#FCD34D", zorder=0)
ax.set_title("C. Year-on-Year Change in Employment Gap (pp)", fontweight="bold")
ax.set_ylabel("Change in gap (pp)")
ax.set_xlabel("Year")
ax.set_xticks(yoy["YEAR"])
ax.tick_params(axis="x", rotation=45)
red_patch   = mpatches.Patch(color=C_NW,   alpha=0.85, label="Gap widens (red)")
green_patch = mpatches.Patch(color=C_CTRL, alpha=0.85, label="Gap narrows (green)")
ax.legend(handles=[red_patch, green_patch], fontsize=8.5)

plt.tight_layout()
fig.savefig(BASE / "trend_analysis_panels.png",
            dpi=160, bbox_inches="tight")
print(f"Figure saved: {BASE / 'trend_analysis_panels.png'}")
plt.close(fig)
# ══════════════════════════════════════════════════════════════
# 5. FIGURE 2: Episode deep-dive
# ══════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Deep-Dive: Three Key Episodes Driving the Employment Gap Trend",
              fontsize=13, fontweight="bold")

# (i) Syrian crisis
ax = axes2[0]
ep_years = list(range(2013, 2019))
ep = annual[annual.YEAR.isin(ep_years)]
ax.plot(ep["YEAR"], ep["native_emp"], color=C_NAT, lw=2.5, marker="o", label="Native")
ax.plot(ep["YEAR"], ep["nw_emp"],     color=C_NW,  lw=2.5, marker="s", label="NW immigrant")
ax.fill_between(ep["YEAR"], ep["nw_emp"], ep["native_emp"], alpha=0.1, color=C_GAP)
ax.axvspan(2015, 2016, alpha=0.25, color="#FCD34D")
ax.annotate("", xy=(2014.5, 50), xytext=(2016.5, 50),
            arrowprops=dict(arrowstyle="<->", color="#92400E", lw=1.5))
ax.text(2015.5, 49.2, "Crisis\nwindow", ha="center", fontsize=8.5, color="#92400E", fontweight="bold")
ax.set_title("i. Syrian Refugee Crisis\n(2015–2016 mass arrival)", fontweight="bold")
ax.set_ylabel("Employment rate (%)")
ax.set_xlabel("Year")
ax.set_ylim(45, 82)
ax.set_xticks(ep_years)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# annotate gap values
for yr in ep_years:
    gap = val("emp_gap", yr)
    nw  = val("nw_emp",  yr)
    ax.annotate(f"{gap:.1f}pp", xy=(yr, nw - 1.5), ha="center", fontsize=7.5, color=C_GAP)

# (ii) COVID-19
ax = axes2[1]
ep_years2 = list(range(2018, 2023))
ep2 = annual[annual.YEAR.isin(ep_years2)]
ax.plot(ep2["YEAR"], ep2["native_emp"], color=C_NAT, lw=2.5, marker="o", label="Native")
ax.plot(ep2["YEAR"], ep2["nw_emp"],     color=C_NW,  lw=2.5, marker="s", label="NW immigrant")
ax.fill_between(ep2["YEAR"], ep2["nw_emp"], ep2["native_emp"], alpha=0.1, color=C_GAP)
ax.axvspan(2020, 2021, alpha=0.25, color="#93C5FD")
ax.text(2020, 78.5, "COVID-19\n2020", ha="center", fontsize=8.5, color="#1D4ED8", fontweight="bold")
ax.set_title("ii. COVID-19 Pandemic Shock\n(2020)", fontweight="bold")
ax.set_ylabel("Employment rate (%)")
ax.set_xlabel("Year")
ax.set_ylim(55, 85)
ax.set_xticks(ep_years2)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for yr in ep_years2:
    gap = val("emp_gap", yr)
    nw  = val("nw_emp",  yr)
    ax.annotate(f"{gap:.1f}pp", xy=(yr, nw - 1.5), ha="center", fontsize=7.5, color=C_GAP)

# (iii) Long-run narrowing
ax = axes2[2]
ax.plot(YEARS, annual["native_emp"], color=C_NAT, lw=2.5, marker="o", label="Native")
ax.plot(YEARS, annual["nw_emp"],     color=C_NW,  lw=2.5, marker="s", label="NW immigrant")
ax.fill_between(YEARS, annual["nw_emp"], annual["native_emp"], alpha=0.12, color=C_GAP, label="Employment gap")
ax.axvspan(2015, 2016, alpha=0.18, color="#FCD34D")
ax.axvspan(2020, 2021, alpha=0.18, color="#93C5FD")
# Trend arrow for narrowing
ax.annotate("", xy=(2022, 67), xytext=(2016, 50),
            arrowprops=dict(arrowstyle="->", color=C_CTRL, lw=2, connectionstyle="arc3,rad=-0.2"))
ax.text(2020, 55, f"NW gains\n+{nw_2022-nw_2010:.0f}pp\nin 12 yrs", ha="center",
        fontsize=8.5, color=C_CTRL, fontweight="bold")
ax.set_title("iii. Long-Run Narrowing\n(2010–2022: −12.5 pp gap)", fontweight="bold")
ax.set_ylabel("Employment rate (%)")
ax.set_xlabel("Year")
ax.set_ylim(38, 88)
ax.set_xticks(YEARS)
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig2.savefig(BASE / "trend_analysis_episodes.png", dpi=160, bbox_inches="tight")
print(f"Figure saved: {BASE / 'trend_analysis_episodes.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════
# 6. Export trend tables to Excel
# ══════════════════════════════════════════════════════════════
trend_by_treat_full = (df.groupby(["YEAR","TREAT"])
    .agg(emp_gap    = ("EMP_GAP","mean"),
         unemp_gap  = ("UNEMP_GAP","mean"),
         native_emp = ("NATIVE_EMP_RATE","mean"),
         nw_emp     = ("NW_EMP_RATE","mean"),
         native_un  = ("NATIVE_UNEMP","mean"),
         nw_un      = ("NW_UNEMP","mean"),
         nw_share   = ("NW_IMMIG_SHARE","mean"),
         sh_share   = ("SH_SHARE","mean"))
    .round(3).reset_index())

trend_by_treat_full["Group"] = trend_by_treat_full["TREAT"].map({1:"Treated",0:"Control"})

with pd.ExcelWriter(BASE / "trend_analysis.xlsx", engine="openpyxl") as w:
    annual.round(3).to_excel(w, sheet_name="Annual All", index=False)
    trend_by_treat_full.to_excel(w, sheet_name="By Group", index=False)

print(f"Excel saved: {BASE / 'trend_analysis.xlsx'}")
print("\nDone.")
