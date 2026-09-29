"""Execute the tutorial notebook and compile the technical report from its actual outputs.

    python tools/build_report.py

Steps: (1) execute the notebook defined in build_tutorial_notebook.py and save it with outputs;
(2) export every figure and the key numbers; (3) write report/results.tex; (4) run pdflatex.
Needs nbformat, nbclient, ipykernel and a LaTeX installation (pdflatex).
"""
import base64
import json
import shutil
import subprocess
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_notebook

from build_tutorial_notebook import OUT as NOTEBOOK, cells

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"

EXPORT = r'''
import json as _json
_f = lambda a: [float(x) for x in np.ravel(a)]
print("@@METRICS@@" + _json.dumps(dict(
    explained=_f(family.explained), sigma=float(family.sigma), scale=_f(family.scale),
    r2_whitened=_f(r_squared(family.z_train, hidden)), r2_plain=_f(r_squared(naive.z_train, hidden)),
    scratch_relerr={str(T): _f(np.median(np.abs(estimates[T] - test["eta"]) / test["eta"], axis=0)) for T in estimates},
    example_rmse={k: float(v) for k, v in example_rmse.items()},
    budgets=list(budgets),
    budget_median={m: _f(np.median(budget_err[m], axis=1)) for m in methods},
    budget_q25={m: _f(np.percentile(budget_err[m], 25, axis=1)) for m in methods},
    budget_q75={m: _f(np.percentile(budget_err[m], 75, axis=1)) for m in methods},
    score_motion_corr=float(np.corrcoef(motion, scores)[0, 1]),
    posterior_det={k: float(v) for k, v in posterior_det.items()},
    design_budgets=list(design_budgets),
    design_mean={m: _f(design_err[m].mean(axis=1)) for m in design_err},
    design_stats=[dict(reduction=float(d["reduction"]), win=float(d["win"]), ci=_f(d["ci"])) for d in design_stats],
    shifts=_f(shifts), threshold=float(threshold), direction=_f(direction),
    in_family_flag_rate=float(np.mean(np.asarray(in_family) > threshold)),
    shift_median={m: _f(np.median(shift_err[m], axis=1)) for m in shift_err},
    flagged=_f(flagged.mean(axis=1)),
    ctrl_mean={m: float(np.mean(v)) for m, v in ctrl_err.items()},
    ctrl_std={m: float(np.std(v)) for m, v in ctrl_err.items()},
)))
'''


def execute():
    nb = new_notebook(cells=[nbformat.from_dict(c) for c in cells] + [new_code_cell(EXPORT)],
                      metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                "language_info": {"name": "python"}})
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    export_cell = nb.cells.pop()
    nbformat.write(nb, NOTEBOOK)
    text = "".join(o.get("text", "") for o in export_cell.outputs)
    metrics = json.loads(text.split("@@METRICS@@", 1)[1])
    FIGURES.mkdir(parents=True, exist_ok=True)
    n = 0
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if "image/png" in out.get("data", {}):
                n += 1
                (FIGURES / f"fig{n:02d}.png").write_bytes(base64.b64decode(out["data"]["image/png"]))
    metrics["n_figures"] = n
    (REPORT / "tutorial_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def fmt(x, d=4):
    return f"{x:.{d}f}"


def write_results_tex(m):
    mac = {}
    mac["ExplOne"], mac["ExplTwo"] = fmt(100 * m["explained"][0], 1), fmt(100 * m["explained"][1], 1)
    mac["ExplRankTwo"] = fmt(100 * sum(m["explained"][:2]), 1)
    mac["SigmaHat"] = fmt(m["sigma"])
    mac["RtwoPlainOne"], mac["RtwoPlainTwo"] = (fmt(x, 2) for x in m["r2_plain"])
    mac["RtwoWhiteOne"], mac["RtwoWhiteTwo"] = (fmt(x, 2) for x in m["r2_whitened"])
    mac["ScaleVec"] = ", ".join(fmt(x, 4) for x in m["scale"])
    mac["ScoreMotionCorr"] = fmt(m["score_motion_corr"], 2)
    mac["Threshold"] = fmt(m["threshold"])
    mac["InFamilyFlagRate"] = fmt(100 * m["in_family_flag_rate"], 0)
    mac["Direction"] = ", ".join(fmt(x, 2) for x in m["direction"])
    for key, name in zip(["no adaptation (average system)", "from scratch (4 coefficients)",
                          "pretrained + adapted context (2 numbers)"], ["ExNone", "ExScratch", "ExContext"]):
        mac[name] = fmt(m["example_rmse"][key])
    for key, name in (("worst", "DetWorst"), ("random", "DetRandom"), ("designed", "DetDesigned")):
        mac[name] = f"{m['posterior_det'][key]:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}"
    for key, name in (("none", "CtrlNone"), ("context", "CtrlContext"), ("reference", "CtrlRef")):
        mac[name] = f"{fmt(m['ctrl_mean'][key])} \\pm {fmt(m['ctrl_std'][key])}"

    rows = []
    for T in ("40", "400"):
        rel = m["scratch_relerr"][T]
        rows.append(f"{int(T) // 10} s & " + " & ".join(f"{100 * r:.1f}\\,\\%" for r in rel) + r" \\")
    mac["ScratchRows"] = "\n".join(rows)

    rows = []
    for i, T in enumerate(m["budgets"]):
        cells_ = []
        for meth in ("none", "scratch", "context", "reference"):
            cells_.append(f"{fmt(m['budget_median'][meth][i])} "
                          f"{{\\scriptsize[{fmt(m['budget_q25'][meth][i], 3)}, {fmt(m['budget_q75'][meth][i], 3)}]}}")
        ratio = m["budget_median"]["scratch"][i] / m["budget_median"]["context"][i]
        rows.append(f"{T / 10:g} & " + " & ".join(cells_) + f" & {ratio:.1f} \\\\")
    mac["BudgetRows"] = "\n".join(rows)

    rows = []
    for i, T in enumerate(m["design_budgets"]):
        st = m["design_stats"][i]
        rows.append(f"{T / 10:g} & {fmt(m['design_mean']['worst'][i])} & {fmt(m['design_mean']['context'][i])} & "
                    f"{fmt(m['design_mean']['designed'][i])} & {100 * st['reduction']:+.1f}\\,\\% & "
                    f"{100 * st['win']:.0f}\\,\\% & $[{st['ci'][0]:+.4f},\\ {st['ci'][1]:+.4f}]$ \\\\")
    mac["DesignRows"] = "\n".join(rows)

    rows = []
    for i, d in enumerate(m["shifts"]):
        rows.append(f"{d:.2f} & {fmt(m['shift_median']['scratch'][i])} & {fmt(m['shift_median']['context'][i])} & "
                    f"{100 * m['flagged'][i]:.0f}\\,\\% \\\\")
    mac["ShiftRows"] = "\n".join(rows)

    tex = ["% Auto-generated by tools/build_report.py from the executed notebook. Do not edit."]
    tex += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    (REPORT / "results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def compile_pdf():
    exe = shutil.which("pdflatex") or r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
    for _ in range(2):
        subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", "tutorial_report.tex"],
                       cwd=REPORT, check=True, stdout=subprocess.DEVNULL)
    print(f"wrote {REPORT / 'tutorial_report.pdf'}")


if __name__ == "__main__":
    metrics = execute()
    assert metrics["n_figures"] == 11, f"expected 11 figures, found {metrics['n_figures']}"
    write_results_tex(metrics)
    compile_pdf()
