from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_XLSX = Path(r"D:\BaiduSyncdisk\CMML\oxydata dap cells.xlsx")
SELECTED_RECORDINGS = ["MAL11E", "CBA1R8C1"]
HEADER_ROWS = 5


def longest_monotonic_segment(times_s: np.ndarray) -> np.ndarray:
    if len(times_s) < 2:
        return times_s

    split_points = np.where(np.diff(times_s) < 0)[0] + 1
    segments = np.split(times_s, split_points)
    return max(segments, key=len)


def spike_stats(times_s: np.ndarray) -> dict[str, float]:
    times_ms = times_s * 1000.0
    isi = np.diff(times_ms)

    freq_hz = 1000.0 / isi.mean()
    frac_0_20 = np.mean((isi >= 0) & (isi < 20))
    frac_20_80 = np.mean((isi >= 20) & (isi < 80))
    frac_80_200 = np.mean((isi >= 80) & (isi < 200))

    hist_bins_5 = np.arange(0, 605, 5)
    hist_counts_5, edges_5 = np.histogram(isi[(isi >= 0) & (isi < 600)], bins=hist_bins_5)
    hist_norm_5 = hist_counts_5 / max(len(isi), 1) * 10000.0

    # Match the in-app hazard calculation as closely as possible.
    histsize = 20000
    hist1 = np.zeros(histsize + 1, dtype=float)
    for interval in isi:
        index = int(interval)
        if 0 <= index < histsize:
            hist1[index] += 1

    haz1 = np.zeros_like(hist1)
    hazcount = 0.0
    spikecount = len(isi) + 1
    for i in range(len(hist1)):
        denom = spikecount - hazcount
        haz1[i] = hist1[i] / denom if denom > 0 else 0.0
        hazcount += hist1[i]

    haz5 = np.zeros((histsize // 5) + 1, dtype=float)
    for i in range(len(hist1)):
        haz5[i // 5] += haz1[i]

    def mean_hazard(start_ms: int, end_ms: int) -> float:
        return float(np.mean(haz5[start_ms // 5:end_ms // 5]))

    return {
        "spike_count": float(len(times_s)),
        "duration_s": float(times_s[-1] - times_s[0]),
        "freq_hz": float(freq_hz),
        "frac_0_20ms": float(frac_0_20),
        "frac_20_80ms": float(frac_20_80),
        "frac_80_200ms": float(frac_80_200),
        "haz_0_20ms": mean_hazard(0, 20),
        "haz_20_80ms": mean_hazard(20, 80),
        "haz_80_200ms": mean_hazard(80, 200),
        "haz_200_400ms": mean_hazard(200, 400),
        "hist_x_5ms": edges_5[:-1],
        "hist_norm_5ms": hist_norm_5,
        "haz_x_5ms": np.arange(len(haz5)) * 5,
        "haz_5ms": haz5,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    analysis_dir = repo_root / "analysis"
    data_dir.mkdir(exist_ok=True)
    analysis_dir.mkdir(exist_ok=True)

    raw = pd.read_excel(SOURCE_XLSX)

    export_columns: dict[str, pd.Series] = {}
    metrics_rows: list[dict[str, float | str]] = []
    plot_payload: dict[str, dict[str, float | np.ndarray]] = {}

    for recording in SELECTED_RECORDINGS:
        series = raw[recording]
        metadata = series.iloc[:HEADER_ROWS].tolist()
        numeric_times = pd.to_numeric(series.iloc[HEADER_ROWS:], errors="coerce").dropna().to_numpy(dtype=float)
        clean_times = longest_monotonic_segment(numeric_times)

        export_values = metadata + clean_times.tolist()
        export_columns[recording] = pd.Series(export_values)

        stats = spike_stats(clean_times)
        plot_payload[recording] = stats
        metrics_rows.append(
            {
                "recording": recording,
                "spike_count": int(stats["spike_count"]),
                "duration_s": stats["duration_s"],
                "freq_hz": stats["freq_hz"],
                "frac_0_20ms": stats["frac_0_20ms"],
                "frac_20_80ms": stats["frac_20_80ms"],
                "frac_80_200ms": stats["frac_80_200ms"],
                "haz_0_20ms": stats["haz_0_20ms"],
                "haz_20_80ms": stats["haz_20_80ms"],
                "haz_80_200ms": stats["haz_80_200ms"],
                "haz_200_400ms": stats["haz_200_400ms"],
            }
        )

    export_df = pd.DataFrame(export_columns)
    export_csv = data_dir / "selected_dap_recordings.csv"
    export_df.to_csv(export_csv, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = analysis_dir / "selected_dap_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for recording in SELECTED_RECORDINGS:
        stats = plot_payload[recording]
        axes[0].plot(stats["hist_x_5ms"], stats["hist_norm_5ms"], lw=2, label=recording)
        axes[1].plot(stats["haz_x_5ms"][:120], stats["haz_5ms"][:120], lw=2, label=recording)

    for axis in axes:
        axis.axvspan(20, 80, color="gold", alpha=0.18)
        axis.set_xlim(0, 600)
        axis.grid(alpha=0.3)
        axis.legend()

    axes[0].set_title("Selected DAP Recordings: ISI Histogram 5 ms Norm")
    axes[1].set_title("Selected DAP Recordings: Hazard 5 ms")
    axes[0].set_ylabel("Norm Count")
    axes[1].set_ylabel("Hazard")
    axes[1].set_xlabel("ISI (ms)")

    overlay_png = analysis_dir / "selected_dap_overlay.png"
    fig.savefig(overlay_png, dpi=160)

    print(f"wrote {export_csv}")
    print(f"wrote {metrics_csv}")
    print(f"wrote {overlay_png}")


if __name__ == "__main__":
    main()
