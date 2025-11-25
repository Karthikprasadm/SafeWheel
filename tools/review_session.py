#!/usr/bin/env python
"""
Generate quick-look charts and an HTML report from events_log.csv plus incident captures.
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review SafeWheel sessions.")
    parser.add_argument("--log", type=Path, default=Path("events_log.csv"),
                        help="Path to events_log.csv (default: ./events_log.csv)")
    parser.add_argument("--incidents", type=Path, default=Path("incidents"),
                        help="Directory containing incident_* folders (default: ./incidents)")
    parser.add_argument("--output", type=Path, default=Path("reports"),
                        help="Directory to write charts/report into (default: ./reports)")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="Open the resulting HTML report in your default browser.")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def format_float(value: Optional[float], suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.2f}{suffix}"


def build_summary(metrics: pd.DataFrame) -> dict:
    summary = {}
    if metrics.empty:
        summary.update({
            "samples": 0,
            "max_debt": None,
            "avg_debt": None,
            "avg_perclos": None,
            "avg_score": None
        })
        return summary
    summary["samples"] = int(metrics.shape[0])
    for column, key in [("debt_score", "debt"), ("perclos", "perclos"), ("score", "score")]:
        series = metrics[column].dropna()
        if series.empty:
            summary[f"avg_{key}"] = None
            if key == "debt":
                summary["max_debt"] = None
        else:
            summary[f"avg_{key}"] = float(series.mean())
            if key == "debt":
                summary["max_debt"] = float(series.max())
    return summary


def render_chart(metrics: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if metrics.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(metrics["timestamp"], metrics["perclos"], color="#1f77b4")
    axes[0].set_ylabel("PERCLOS")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.2)
    axes[1].plot(metrics["timestamp"], metrics["debt_score"], color="#d62728")
    axes[1].set_ylabel("Debt")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.2)
    axes[1].set_xlabel("Time")
    fig.suptitle("Session Metrics")
    fig.tight_layout()
    chart_path = output_path / "session_metrics.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def list_incidents(incidents_dir: Path) -> list:
    if not incidents_dir.exists():
        return []
    folders = sorted(p for p in incidents_dir.iterdir() if p.is_dir() and p.name.startswith("incident_"))
    results = []
    for folder in folders:
        results.append({
            "name": folder.name,
            "snapshot": folder / "snapshot.jpg",
            "video": folder / "clip.mp4"
        })
    return results


def generate_report(metrics: pd.DataFrame, summary: dict, incidents: list,
                    chart_path: Optional[Path], output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"session_report_{timestamp}.html"
    html = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append("<title>SafeWheel Session Report</title>")
    html.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:2rem;} "
                "table{border-collapse:collapse;margin-bottom:1.5rem;} "
                "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;} "
                "h2{margin-top:2.5rem;} figure{margin:1rem 0;}</style></head><body>")
    html.append("<h1>SafeWheel Session Report</h1>")
    html.append("<h2>Summary</h2>")
    html.append("<table><tr><th>Metric</th><th>Value</th></tr>")
    html.append(f"<tr><td>Metrics samples</td><td>{summary.get('samples', 0)}</td></tr>")
    html.append(f"<tr><td>Avg Debt</td><td>{format_float(summary.get('avg_debt'))}</td></tr>")
    html.append(f"<tr><td>Max Debt</td><td>{format_float(summary.get('max_debt'))}</td></tr>")
    html.append(f"<tr><td>Avg PERCLOS</td><td>{format_float(summary.get('avg_perclos'))}</td></tr>")
    html.append(f"<tr><td>Avg Score</td><td>{format_float(summary.get('avg_score'))}</td></tr>")
    html.append("</table>")
    if chart_path:
        rel_chart = os.path.relpath(chart_path, report_path.parent)
        html.append("<h2>Time Series</h2>")
        html.append("<figure>")
        html.append(f"<img src='{rel_chart}' alt='Session metrics chart' style='max-width:100%;height:auto;'/>")
        html.append("</figure>")
    html.append("<h2>Incidents</h2>")
    if not incidents:
        html.append("<p>No incident folders were found.</p>")
    else:
        for incident in incidents:
            html.append(f"<h3>{incident['name']}</h3>")
            snapshot = incident["snapshot"]
            video = incident["video"]
            if snapshot.exists():
                rel_snap = os.path.relpath(snapshot, report_path.parent)
                html.append(f"<p><img src='{rel_snap}' alt='Snapshot' style='max-width:320px;border:1px solid #ccc;'/></p>")
            if video.exists():
                rel_video = os.path.relpath(video, report_path.parent)
                html.append(
                    "<video width='480' controls preload='metadata'>"
                    f"<source src='{rel_video}' type='video/mp4'>"
                    "Your browser does not support the video tag."
                    "</video>"
                )
    html.append("</body></html>")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return report_path


def main() -> int:
    args = parse_args()
    if not args.log.exists():
        print(f"[ERROR] Log file not found: {args.log}", file=sys.stderr)
        return 1
    try:
        df = pd.read_csv(args.log)
    except Exception as exc:
        print(f"[ERROR] Failed to read {args.log}: {exc}", file=sys.stderr)
        return 1
    if "timestamp" not in df.columns:
        print("[ERROR] events_log.csv must contain a 'timestamp' column.", file=sys.stderr)
        return 1
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = ensure_columns(df, ("score", "perclos", "debt_score", "yawns_per_min"))
    metrics = df[df["event"] == "metrics"].copy()
    metrics.sort_values("timestamp", inplace=True)

    args.output.mkdir(parents=True, exist_ok=True)
    chart_path = render_chart(metrics, args.output)
    summary = build_summary(metrics)
    incidents = list_incidents(args.incidents)
    report_path = generate_report(metrics, summary, incidents, chart_path, args.output)
    print(f"[INFO] Report written to {report_path}")
    if args.open_browser:
        webbrowser.open(report_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

