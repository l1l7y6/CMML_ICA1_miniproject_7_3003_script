from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import wx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spikemod import SecData, SpikeModel, StateData
from HypoModPy.hypospikes import NeuroDat, SpikeDat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


wx.QueueEvent = lambda *args, **kwargs: None

SELECTED_RECORDINGS_CSV = Path("data/selected_dap_recordings.csv")
OUTPUT_DIR = Path("analysis/baseline_fit")
HEADER_ROWS = 5

DEFAULT_SPIKE_PARAMS = {
    "runtime": 240,
    "hstep": 1,
    "Vrest": -62,
    "Vthresh": -50,
    "psprate": 300,
    "pspratio": 1,
    "pspmag": 3,
    "halflifeMem": 7.5,
    "kHAP": 60,
    "halflifeHAP": 8,
    "kAHP": 0.5,
    "halflifeAHP": 500,
    "kDAP": 0.0,
    "halflifeDAP": 80,
    "useNMDA": 0,
    "kNMDA": 0.0,
    "halflifeNMDARise": 8.0,
    "halflifeNMDADecay": 120.0,
}

DEFAULT_SEC_PARAMS = {
    "kB": 0.021,
    "halflifeB": 2000,
    "Bbase": 0.5,
    "kC": 0.0003,
    "halflifeC": 20000,
    "kE": 1.5,
    "halflifeE": 100,
    "Cth": 0.14,
    "Cgradient": 5,
    "Eth": 12,
    "Egradient": 5,
    "beta": 120,
    "Rmax": 1000000,
    "Rinit": 1000000,
    "Pmax": 5000,
    "alpha": 0.003,
    "plasma_hstep": 1,
    "halflifeDiff": 61,
    "halflifeClear": 68,
    "VolPlasma": 100,
    "VolEVF": 9.75,
    "secExp": 2,
}


class DummyScaleBox:
    def GraphUpdateAll(self) -> None:
        pass


class DummyMainWin:
    def __init__(self) -> None:
        self.scalebox = DummyScaleBox()


class DummySpikeBox:
    def __init__(self) -> None:
        self.modflags = {"randomflag": 1}


class DummyMod:
    def __init__(self) -> None:
        self.datsample = 1
        self.secdata = SecData(1000000)
        self.statedata = StateData(1000000)
        self.modspike = SpikeDat()
        self.spikebox = DummySpikeBox()
        self.mainwin = DummyMainWin()


@dataclass
class TargetRecording:
    name: str
    metadata: list[str]
    times_s: np.ndarray
    analysis: SpikeDat


def load_targets() -> list[TargetRecording]:
    raw = pd.read_csv(SELECTED_RECORDINGS_CSV)
    targets: list[TargetRecording] = []

    for name in raw.columns:
        column = raw[name]
        metadata = ["" if pd.isna(x) else str(x) for x in column.iloc[:HEADER_ROWS].tolist()]
        times_s = pd.to_numeric(column.iloc[HEADER_ROWS:], errors="coerce").dropna().to_numpy(dtype=float)
        targets.append(TargetRecording(name=name, metadata=metadata, times_s=times_s, analysis=analyze_times(name, times_s)))

    return targets


def analyze_times(name: str, times_s: np.ndarray) -> SpikeDat:
    neuro = NeuroDat()
    neuro.name = name
    neuro.spikecount = len(times_s)
    neuro.times[: len(times_s)] = times_s * 1000.0

    spike = SpikeDat()
    spike.Analysis(neuro)
    spike.name = name
    return spike


def run_model(spike_params: dict[str, float], sec_params: dict[str, float]) -> SpikeDat:
    mod = DummyMod()
    model = SpikeModel(mod, {"spike": spike_params, "sec": sec_params})
    model.Model()
    mod.modspike.Analysis()
    return mod.modspike


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def normalised_rmse(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(b))), 1e-9)
    return rmse(a, b) / scale


def score_stage1(model: SpikeDat, target: SpikeDat) -> float:
    hist_target = np.asarray(target.hist5norm[:40], dtype=float)
    hist_model = np.asarray(model.hist5norm[:40], dtype=float)
    haz_target = np.asarray(target.haz5[:40], dtype=float)
    haz_model = np.asarray(model.haz5[:40], dtype=float)
    freq_err = abs(model.freq - target.freq) / max(target.freq, 1e-9)
    hist_err = normalised_rmse(hist_model, hist_target)
    haz_err = normalised_rmse(haz_model, haz_target)
    return 0.40 * freq_err + 0.35 * hist_err + 0.25 * haz_err


def score_stage2(model: SpikeDat, target: SpikeDat) -> float:
    hist_target = np.asarray(target.hist5norm[:40], dtype=float)
    hist_model = np.asarray(model.hist5norm[:40], dtype=float)
    haz_target = np.asarray(target.haz5[:40], dtype=float)
    haz_model = np.asarray(model.haz5[:40], dtype=float)
    iod_target = np.asarray(target.IoDdata[:7], dtype=float)
    iod_model = np.asarray(model.IoDdata[:7], dtype=float)
    freq_err = abs(model.freq - target.freq) / max(target.freq, 1e-9)
    hist_err = normalised_rmse(hist_model, hist_target)
    haz_err = normalised_rmse(haz_model, haz_target)
    iod_err = normalised_rmse(iod_model, iod_target)
    return 0.25 * freq_err + 0.30 * hist_err + 0.20 * haz_err + 0.25 * iod_err


def candidate_params(target: TargetRecording, runtime: int) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    if target.analysis.freq >= 5.0:
        psprates = [280, 340, 400, 460, 520]
    else:
        psprates = [220, 280, 340, 400, 460]

    stage1 = []
    for psprate, k_hap, hl_hap in itertools.product(psprates, [50, 70, 90], [8, 12, 16]):
        params = dict(DEFAULT_SPIKE_PARAMS)
        params.update(
            {
                "runtime": runtime,
                "psprate": psprate,
                "kHAP": k_hap,
                "halflifeHAP": hl_hap,
                "kAHP": 0.0,
                "halflifeAHP": 500,
            }
        )
        stage1.append(params)

    stage2 = []
    for k_ahp, hl_ahp in itertools.product([0.0, 0.15, 0.30, 0.45, 0.60], [300, 500, 700]):
        params = {"kAHP": k_ahp, "halflifeAHP": hl_ahp}
        stage2.append(params)

    return stage1, stage2


def fit_recording(target: TargetRecording) -> dict[str, object]:
    stage1_runtime = 240
    final_runtime = 1200
    stage1_candidates, stage2_adjustments = candidate_params(target, stage1_runtime)

    stage1_results: list[dict[str, object]] = []
    for params in stage1_candidates:
        model = run_model(params, DEFAULT_SEC_PARAMS)
        stage1_results.append({"score": score_stage1(model, target.analysis), "params": params, "model": model})
    stage1_results.sort(key=lambda item: float(item["score"]))

    stage2_results: list[dict[str, object]] = []
    for base in stage1_results[:4]:
        for adjust in stage2_adjustments:
            params = dict(base["params"])
            params.update(adjust)
            model = run_model(params, DEFAULT_SEC_PARAMS)
            stage2_results.append({"score": score_stage2(model, target.analysis), "params": params, "model": model})
    stage2_results.sort(key=lambda item: float(item["score"]))

    top_final: list[dict[str, object]] = []
    for candidate in stage2_results[:3]:
        params = dict(candidate["params"])
        params["runtime"] = final_runtime
        model = run_model(params, DEFAULT_SEC_PARAMS)
        top_final.append({"score": score_stage2(model, target.analysis), "params": params, "model": model})
    top_final.sort(key=lambda item: float(item["score"]))

    best = top_final[0]
    best_model = best["model"]
    best_params = best["params"]

    return {
        "target": target,
        "best_score": float(best["score"]),
        "best_params": best_params,
        "best_model": best_model,
        "stage1_top_scores": [float(row["score"]) for row in stage1_results[:5]],
        "stage2_top_scores": [float(row["score"]) for row in stage2_results[:5]],
    }


def save_comparison_plot(target: TargetRecording, best_model: SpikeDat, best_params: dict[str, float]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    x_hist = np.arange(120) * 5
    axes[0].plot(x_hist, np.asarray(target.analysis.hist5norm[:120]), lw=2, label=f"{target.name} target")
    axes[0].plot(x_hist, np.asarray(best_model.hist5norm[:120]), lw=2, label="baseline model")
    axes[0].axvspan(20, 80, color="gold", alpha=0.18)
    axes[0].set_title(f"{target.name}: Hist 5 ms Norm")
    axes[0].set_ylabel("Norm Count")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    x_haz = np.arange(120) * 5
    axes[1].plot(x_haz, np.asarray(target.analysis.haz5[:120]), lw=2, label=f"{target.name} target")
    axes[1].plot(x_haz, np.asarray(best_model.haz5[:120]), lw=2, label="baseline model")
    axes[1].axvspan(20, 80, color="gold", alpha=0.18)
    axes[1].set_title("Hazard 5 ms")
    axes[1].set_ylabel("Hazard")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    iod_x = np.asarray(target.analysis.IoDdataX[:7])
    axes[2].plot(iod_x, np.asarray(target.analysis.IoDdata[:7]), marker="o", lw=2, label=f"{target.name} target")
    axes[2].plot(iod_x, np.asarray(best_model.IoDdata[:7]), marker="o", lw=2, label="baseline model")
    axes[2].set_title(
        "IoD Range"
        + f"\npsprate={best_params['psprate']}, kHAP={best_params['kHAP']}, hlHAP={best_params['halflifeHAP']}, "
        + f"kAHP={best_params['kAHP']}, hlAHP={best_params['halflifeAHP']}"
    )
    axes[2].set_xlabel("IoD window index")
    axes[2].set_ylabel("IoD")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    out = OUTPUT_DIR / f"{target.name.lower()}_baseline_compare.png"
    fig.savefig(out, dpi=160)
    return out


def main() -> None:
    targets = load_targets()
    summaries = []

    for target in targets:
        print(f"Fitting baseline for {target.name}...")
        result = fit_recording(target)
        best_model = result["best_model"]
        best_params = result["best_params"]
        plot_path = save_comparison_plot(target, best_model, best_params)

        summary = {
            "recording": target.name,
            "target_freq_hz": float(target.analysis.freq),
            "model_freq_hz": float(best_model.freq),
            "best_score": float(result["best_score"]),
            "psprate": float(best_params["psprate"]),
            "kHAP": float(best_params["kHAP"]),
            "halflifeHAP": float(best_params["halflifeHAP"]),
            "kAHP": float(best_params["kAHP"]),
            "halflifeAHP": float(best_params["halflifeAHP"]),
            "runtime_s": float(best_params["runtime"]),
            "plot_path": str(plot_path),
        }
        summaries.append(summary)

        json_path = OUTPUT_DIR / f"{target.name.lower()}_baseline_fit.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": summary,
                    "stage1_top_scores": result["stage1_top_scores"],
                    "stage2_top_scores": result["stage2_top_scores"],
                },
                handle,
                indent=2,
            )
        print(f"  best params: {summary}")

    summary_df = pd.DataFrame(summaries)
    summary_csv = OUTPUT_DIR / "baseline_fit_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
