"""Turn results/*.csv into figures/*.pdf|png and report/generated/*.tex."""
from common import ROOT, RESULTS, begin, end
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

begin("figures and report tables")
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
GEN = ROOT / "report" / "generated"; GEN.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150,
                     "axes.titlesize": 7, "legend.fontsize": 6, "lines.linewidth": 1.1})
COL, WIDE = 3.3, 7.0


def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write_tex(df, name, colfmt=None):
    colfmt = colfmt or "l" * len(df.columns)
    lines = [r"\begin{tabular}{" + colfmt + "}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(str(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / f"{name}.tex").write_text("\n".join(lines) + "\n")


def name_of(p, q):
    return {(0, 0): "Newton", (1, 1): "(1,1)"}.get((p, q), f"({p},{q})")


# ---- run info macros -------------------------------------------------------
ri = pd.read_csv(RESULTS / "run_info.csv").set_index("key").value
macros = {"cpuname": ri["cpu"], "osname": ri["os"], "pyver": ri["python"],
          "npver": ri["numpy"], "spver": ri["scipy"], "repeats": ri["timing_repeats"]}
(GEN / "run_info.tex").write_text(
    "".join(f"\\newcommand{{\\{k}}}{{{esc(v)}}}\n" for k, v in macros.items()))

# ---- validation table ------------------------------------------------------
v = pd.read_csv(RESULTS / "validation.csv")
t = pd.DataFrame({"case": v.case,
                  "Newton": [f"{a} / {b}" for a, b in zip(v.newton, v.newton_paper)],
                  "PM(3)": [f"{a} / {b}" for a, b in zip(v.pm3, v.pm3_paper)],
                  "PM(4) best": [f"{a} / {b}" for a, b in zip(v.pm4_best, v.pm4_paper)]})
write_tex(t, "tab_validation", "lccc")

# ---- COC table -------------------------------------------------------------
c = pd.read_csv(RESULTS / "coc.csv")
ok = c[(c.log10_err < -20) & (c.log10_err > -900)].sort_values("k").groupby(["system", "p", "q"]).tail(1)
t = ok.pivot_table(index=["p", "q"], columns="system", values="coc").round(2).reset_index()
t.insert(2, "theory", [min(p + q + 2, 4) for p, q in zip(t.p, t.q)])
t["p,q"] = [f"({p},{q})" for p, q in zip(t.p, t.q)]
t = t[["p,q", "theory", "1a", "1b", "1g"]]
t.columns = ["(p,q)", "theory", "1a", "1b", "1g"]
for col in ["1a", "1b", "1g"]:
    t[col] = t[col].map(lambda x: f"{x:.2f}")
write_tex(t, "tab_coc", "lcccc")

# ---- efficiency ------------------------------------------------------------
e = pd.concat([pd.read_csv(RESULTS / "efficiency.csv"), pd.read_csv(RESULTS / "efficiency_large.csv")],
              ignore_index=True)
e["ms"] = e.time_median * 1e3
e.loc[~e.converged, "ms"] = np.nan                  # time to converge only makes sense if it converged
newt = e[(e.p == 0) & (e.q == 0)].set_index(["system", "n"]).ms
e["rel"] = [ms / newt[(s, n)] for ms, s, n in zip(e.ms, e.system, e.n)]

SHOW = [((0, 0, "generic"), "Newton", "k", "-"),
        ((1, 1, "generic"), "(1,1) generic", "C0", "-"), ((1, 1, "cheap"), "(1,1) cheap", "C0", "--"),
        ((2, 0, "generic"), "(2,0) generic", "C3", "-"), ((2, 0, "cheap"), "(2,0) cheap", "C3", "--"),
        ((2, 1, "generic"), "(2,1) generic", "C2", "-")]
fams = list(e.system.unique())
fig, axs = plt.subplots(2, 3, figsize=(WIDE, 3.7), sharex=True)
for ax, fam in zip(axs.flat, fams):
    for (p, q, form), lab, col, ls in SHOW:
        d = e[(e.system == fam) & (e.p == p) & (e.q == q) & (e.form == form)].sort_values("n")
        ax.plot(d.n, d.rel, ls, color=col, marker="o", ms=2.5, label=lab)
    ax.axhline(1, color="grey", lw=.6)
    ax.set_yscale("log"); ax.set_xscale("log"); ax.set_title(fam)
    ax.set_xticks([21, 101, 401, 1601]); ax.set_xticklabels([21, 101, 401, 1601]); ax.minorticks_off()
for ax in axs[1]:
    ax.set_xlabel("n")
for ax in axs[:, 0]:
    ax.set_ylabel("time / Newton time")
h, l = axs[0, 0].get_legend_handles_labels()
axs.flat[-1].axis("off"); axs.flat[-1].legend(h, l, loc="center", frameon=False)
save(fig, "fig_eff_rel_time")

fig, ax = plt.subplots(figsize=(COL, 2.5))
d = e[e.converged]
for form, col in [("generic", "C0"), ("cheap", "C3")]:
    dd = d[d.form == form]
    ax.loglog(dd.ops_total, dd.ms, ".", color=col, ms=3, label=form)
ax.set_xlabel("modelled operations (iterations x ops per iteration)"); ax.set_ylabel("time [ms]")
ax.legend(); save(fig, "fig_eff_model")

top = e[e.n == 401]
cols = [((0, 0, "generic"), "Newton"), ((2, 0, "generic"), "(2,0) gen."), ((2, 0, "cheap"), "(2,0) chp."),
        ((1, 1, "generic"), "(1,1) gen."), ((1, 1, "cheap"), "(1,1) chp.")]
rows = []
for fam in fams:
    row = {"system": fam}
    for (p, q, form), lab in cols:
        r = top[(top.system == fam) & (top.p == p) & (top.q == q) & (top.form == form)].iloc[0]
        row[lab] = f"{r.iters} / {r.ms:.1f}" if r.converged else f"fails ({r.status})"
    rows.append(row)
tt = pd.DataFrame(rows); tt.columns = ["system"] + [f"{l}" for _, l in cols]
write_tex(tt, "tab_eff", "lccccc")

# large n: Newton against the cheap forms, time relative to Newton
big = e[e.n >= 601]
rows = []
for fam in [f for f in fams if not f.startswith("Trig")]:
    for n in [801, 1601]:
        d = big[(big.system == fam) & (big.n == n)]
        row = {"system": fam, "n": n}
        for (p, q, form), lab in [((0, 0, "generic"), "Newton"), ((1, 0, "cheap"), "(1,0) cheap"),
                                  ((2, 0, "cheap"), "(2,0) cheap"), ((1, 1, "cheap"), "(1,1) cheap")]:
            r = d[(d.p == p) & (d.q == q) & (d.form == form)].iloc[0]
            row[lab] = (f"{r.iters} / {r.ms:.0f}" if lab == "Newton" else f"{r.iters} / {r.ms:.0f} ({r.rel:.2f})")
        rows.append(row)
write_tex(pd.DataFrame(rows), "tab_eff_large", "llcccc")

# ---- robustness ------------------------------------------------------------
r = pd.read_csv(RESULTS / "robustness_summary.csv")
ns = r[r.family == "NearSing"]
sty = {"LU": ("C0", "o-"), "TSVD": ("C1", "s--"), "LM": ("C2", "^:"), "LM-N": ("C3", "v-")}
meths = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]
fig, axs = plt.subplots(2, len(meths), figsize=(WIDE, 3.6), sharex=True)
for j, (p, q) in enumerate(meths):
    for sname, (col, st) in sty.items():
        d = ns[(ns.p == p) & (ns.q == q) & (ns.solver == sname)].sort_values("eps")
        axs[0, j].plot(d.eps, d.accurate_rate, st, color=col, ms=3, label=sname)
        axs[1, j].plot(d.eps, d.med_fwd_error.clip(lower=1e-17), st, color=col, ms=3)
    axs[0, j].set_title(name_of(p, q)); axs[0, j].set_ylim(-0.05, 1.05)
    axs[1, j].set_yscale("log"); axs[1, j].set_xlabel(r"$\epsilon$")
    for a in axs[:, j]:
        a.set_xscale("log")
axs[0, 0].set_ylabel("runs with error < 1e-6"); axs[1, 0].set_ylabel("median forward error")
axs[0, 0].legend(loc="lower left")
save(fig, "fig_rob_nearsing")

ts = pd.read_csv(RESULTS / "tsvd_threshold.csv")
fig, ax = plt.subplots(figsize=(COL, 2.6))
for (eps, col) in zip(sorted(ts.eps.unique()), ["C0", "C1", "C2", "C3"]):
    for (p, q), ls in [((0, 0), "-"), ((1, 1), "--")]:
        d = ts[(ts.eps == eps) & (ts.p == p) & (ts.q == q)].sort_values("rtol")
        ax.plot(d.rtol, d.accurate_rate, ls, color=col, marker="o", ms=2.5,
                label=(f"eps={eps:g}" if (p, q) == (0, 0) else None))
ax.plot([], [], "k-", label="Newton"); ax.plot([], [], "k--", label="(1,1)")
ax.axvline(1e-8, color="grey", lw=.6)
ax.set_xscale("log"); ax.set_xlabel("TSVD threshold (relative to $\\sigma_{max}$)")
ax.set_ylabel("runs with error < 1e-6"); ax.legend(ncol=2)
save(fig, "fig_tsvd_threshold")

nat = r[r.family != "NearSing"]
rows = []
for fam in ["CosDiag", "Powell", "H-eq c=1"]:
    for sname in ["LU", "TSVD", "LM", "LM-N"]:
        row = {"system": fam, "solver": sname}
        for p, q in meths:
            d = nat[(nat.family == fam) & (nat.solver == sname) & (nat.p == p) & (nat.q == q)].iloc[0]
            row[name_of(p, q)] = f"{100 * d.accurate_rate:.0f} ({100 * d.converged_rate:.0f}) / {d.med_iters:.0f}"
        rows.append(row)
write_tex(pd.DataFrame(rows).rename(columns=lambda c: c), "tab_natural", "llccccc")
end()
