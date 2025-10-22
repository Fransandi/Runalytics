#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py — Produce 4 images in ./output from Garmin-like running CSV (data.csv)

Images (in ./output):
1) general_stats.png               — Single table: overall totals + per-week averages
2) race_predictions.png            — Three charts showing 5K/10K/Half/Marathon predictions
3) weekly_trends.png               — Time-series charts for distance, pace, HR trends
4) race_predictions_table.png      — Pretty table with predicted paces & times for all races

Assumptions:
- data.csv exists in the same folder.
- Required columns (subset): Date, Distance, Avg HR, Avg Pace
- Optional time columns: Moving Time / Elapsed Time / Time (hh:mm:ss or mm:ss)
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------- Config ----------
CSV_FILENAME = "data.csv"
OUTPUT_DIR = "output"
LONG_RUN_THRESHOLD_KM = 10.0
RIEGEL_EXPONENT = 1.06  # personalize (1.04–1.09)

# Race distances for predictions
RACE_DISTANCES = {
    "5K": 5.0,
    "10K": 10.0,
    "Half Marathon": 21.0975,
    "Marathon": 42.195
}

# Try a nicer table export tool
_HAVE_DFI = False
try:
    import dataframe_image as dfi
    _HAVE_DFI = True
except Exception:
    _HAVE_DFI = False


# ---------- Utils ----------
def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pace_to_min_per_km(pace_str):
    """Convert 'mm:ss' (or numeric string) to minutes per km as float."""
    if pd.isna(pace_str):
        return None
    s = str(pace_str).strip()
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                m, sec = parts
                return float(m) + float(sec) / 60.0
            elif len(parts) == 3:
                h, m, sec = parts
                return 60 * float(h) + float(m) + float(sec) / 60.0
        return float(s)
    except Exception:
        return None


def duration_to_minutes(x):
    """Convert 'hh:mm:ss' or 'mm:ss' or numeric to minutes."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 3:
                h, m, sec = parts
                return 60 * float(h) + float(m) + float(sec) / 60.0
            elif len(parts) == 2:
                m, sec = parts
                return float(m) + float(sec) / 60.0
        return float(s)
    except Exception:
        return None


def min_per_km_to_str(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    m = int(x)
    s = int(round((x - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}"


def minutes_to_hms_str(total_minutes: float) -> str:
    if total_minutes is None or math.isnan(total_minutes):
        return ""
    h = int(total_minutes // 60)
    m = int(total_minutes % 60)
    s = int((total_minutes * 60) % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def riegel_time_minutes(distance_km, pace_min_per_km, d_goal, exponent=RIEGEL_EXPONENT):
    """Riegel: T2 = T1 * (D2/D1)^exp ; returns minutes."""
    if not distance_km or distance_km <= 0 or pace_min_per_km is None or math.isnan(pace_min_per_km):
        return None
    t1 = pace_min_per_km * distance_km  # minutes
    return t1 * (d_goal / distance_km) ** exponent


def longest_run_each_week(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Week"] = tmp["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    idx = tmp.groupby("Week")["Distance"].idxmax()
    return tmp.loc[idx].sort_values("Date")


def predict_from_subset(subset: pd.DataFrame, label: str):
    """Use MOST RECENT run in subset as reference for Riegel predictions."""
    if subset.empty:
        return {"label": label, "ok": False, "msg": "No rows in subset."}

    ref = subset.iloc[-1]
    d1 = ref["Distance"]
    pace = ref["Avg Pace (min/km)"]

    predictions = {"label": label, "ok": True,
                   "ref_distance": round(float(d1), 2)}

    for race_name, race_distance in RACE_DISTANCES.items():
        t2 = riegel_time_minutes(d1, pace, race_distance)
        if t2 is not None:
            predictions[race_name] = {
                "pace_est": t2 / race_distance,
                "time_min": t2,
                "time_str": minutes_to_hms_str(t2)
            }
        else:
            predictions[race_name] = None

    return predictions


# ---------- Table exporters ----------
def export_table_styled(df: pd.DataFrame, filename: str, caption: str = ""):
    """
    Prefer dataframe_image (dfi) for nicer UI. Fall back to Matplotlib table if dfi unavailable.
    """
    if _HAVE_DFI:
        styled = (
            df.style
            .set_caption(caption)
            .set_table_styles([
                {"selector": "caption",
                 "props": [("text-align", "left"), ("font-size", "16px"), ("font-weight", "bold"),
                           ("padding", "0 0 8px 0")]},
                {"selector": "th",
                 "props": [("background-color", "#f2f2f2"), ("font-weight", "bold"),
                           ("border", "1px solid #ddd"), ("padding", "6px")]},
                {"selector": "td",
                 "props": [("border", "1px solid #eee"), ("padding", "6px")]}
            ])
            .format(precision=2)
            .hide(axis="index")
        )
        dfi.export(styled, filename, table_conversion="matplotlib")
    else:
        # Fallback to a simple Matplotlib table
        fig_width = max(8, min(20, 1 + 0.35 * len(df.columns)))
        fig_height = max(2.8, 1.2 + 0.45 * (len(df) + 2))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        tbl = ax.table(cellText=df.values, colLabels=df.columns,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.25)
        if caption:
            fig.suptitle(caption, fontsize=15, fontweight="bold", y=0.98)
        plt.savefig(filename, dpi=170, bbox_inches="tight")
        plt.close(fig)


# ---------- Charts ----------
def plot_predictions(df, long_runs, weekly_longest, pred_all, pred_long, pred_week, blended, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 17))
    fig.suptitle("Race Pace & Time Predictions (Riegel)",
                 fontsize=18, fontweight="bold")

    # Colors for different races
    race_colors = {"5K": "red", "10K": "orange",
                   "Half Marathon": "blue", "Marathon": "purple"}

    # 1) All runs
    axes[0].scatter(df["Distance"], df["Avg Pace (min/km)"],
                    alpha=0.7, label="All Runs", color="gray")

    if pred_all.get("ok"):
        for race_name, race_data in pred_all.items():
            if race_name in RACE_DISTANCES and race_data:
                axes[0].axhline(race_data["pace_est"], linestyle="--",
                                color=race_colors[race_name],
                                label=f"{race_name}: {min_per_km_to_str(race_data['pace_est'])} ({race_data['time_str']})")

    axes[0].invert_yaxis()
    axes[0].set_title("All Runs → Race Predictions")
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Pace (min/km)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    # 2) Long runs
    axes[1].scatter(long_runs["Distance"], long_runs["Avg Pace (min/km)"],
                    alpha=0.8, label="Long Runs (≥10 km)", color="gray")

    if pred_long.get("ok"):
        for race_name, race_data in pred_long.items():
            if race_name in RACE_DISTANCES and race_data:
                axes[1].axhline(race_data["pace_est"], linestyle="--",
                                color=race_colors[race_name],
                                label=f"{race_name}: {min_per_km_to_str(race_data['pace_est'])} ({race_data['time_str']})")

    axes[1].invert_yaxis()
    axes[1].set_title("Long Runs (≥10 km) → Race Predictions")
    axes[1].set_xlabel("Distance (km)")
    axes[1].set_ylabel("Pace (min/km)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    # 3) Longest run per week
    axes[2].plot(weekly_longest["Date"], weekly_longest["Avg Pace (min/km)"], marker="o",
                 label="Longest Run Pace per Week", color="gray")

    if pred_week.get("ok"):
        for race_name, race_data in pred_week.items():
            if race_name in RACE_DISTANCES and race_data:
                axes[2].axhline(race_data["pace_est"], linestyle="--",
                                color=race_colors[race_name],
                                label=f"{race_name}: {min_per_km_to_str(race_data['pace_est'])} ({race_data['time_str']})")

    axes[2].invert_yaxis()
    axes[2].set_title("Longest Run per Week → Race Predictions")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Pace (min/km)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_weekly_trends(df, out_path):
    """Create weekly trend charts for distance, pace, and heart rate."""
    # Prepare weekly data
    weekly_data = df.copy()
    weekly_data["Week"] = weekly_data["Date"].dt.to_period(
        "W").apply(lambda r: r.start_time)

    # Aggregate by week
    weekly_stats = weekly_data.groupby("Week").agg({
        "Distance": "sum",  # Total distance per week
        "Avg Pace (min/km)": "mean",  # Average pace per week
        "Avg HR": "mean"  # Average heart rate per week
    }).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle("Weekly Training Trends", fontsize=18, fontweight="bold")

    # 1) Distance over time
    axes[0].plot(weekly_stats["Week"], weekly_stats["Distance"],
                 marker="o", linewidth=2, markersize=6, color="blue")
    axes[0].set_title("Weekly Distance", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("Distance (km)")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # 2) Pace over time
    axes[1].plot(weekly_stats["Week"], weekly_stats["Avg Pace (min/km)"],
                 marker="o", linewidth=2, markersize=6, color="green")
    # Faster pace (lower values) should be higher on the chart
    axes[1].invert_yaxis()
    axes[1].set_title("Weekly Average Pace", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("Pace (min/km)")
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    # 3) Heart Rate over time
    axes[2].plot(weekly_stats["Week"], weekly_stats["Avg HR"],
                 marker="o", linewidth=2, markersize=6, color="red")
    axes[2].set_title("Weekly Average Heart Rate",
                      fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Week")
    axes[2].set_ylabel("Heart Rate (bpm)")
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------- Main ----------
def main():
    if not os.path.exists(CSV_FILENAME):
        raise FileNotFoundError(
            f"Could not find '{CSV_FILENAME}' in: {os.getcwd()}")
    ensure_output_dir(OUTPUT_DIR)

    # Load CSV
    raw = pd.read_csv(CSV_FILENAME)

    # Validate columns
    needed = ["Date", "Distance", "Avg HR", "Avg Pace"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}\nFound: {list(raw.columns)}")

    # Clean/convert
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df["Avg HR"] = pd.to_numeric(df["Avg HR"], errors="coerce")
    df["Avg Pace (min/km)"] = df["Avg Pace"].apply(pace_to_min_per_km)

    # Total time column
    time_col = None
    for cand in ["Moving Time", "Elapsed Time", "Time"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col:
        df["Duration (min)"] = df[time_col].apply(duration_to_minutes)
    else:
        df["Duration (min)"] = df["Distance"] * df["Avg Pace (min/km)"]

    # Final clean
    df = df.dropna(subset=["Date", "Distance", "Avg HR",
                   "Avg Pace (min/km)", "Duration (min)"]).sort_values("Date")

    # Subsets
    long_runs = df[df["Distance"] >= LONG_RUN_THRESHOLD_KM].copy()
    weekly_longest = longest_run_each_week(df)

    # Predictions using most recent run in each subset
    pred_all = predict_from_subset(df, "All Runs")
    pred_long = predict_from_subset(long_runs, "Long Runs (≥10 km)")
    pred_week = predict_from_subset(weekly_longest, "Longest Run per Week")

    # Blended result
    valid = [p for p in [pred_all, pred_long, pred_week] if p.get("ok")]
    blended = None
    if valid:
        blended = {"label": "Blended", "ok": True}
        for race_name in RACE_DISTANCES.keys():
            race_preds = [p[race_name] for p in valid if p.get(race_name)]
            if race_preds:
                avg_pace = sum(r["pace_est"]
                               for r in race_preds) / len(race_preds)
                avg_time = sum(r["time_min"]
                               for r in race_preds) / len(race_preds)
                blended[race_name] = {
                    "pace_est": avg_pace,
                    "time_min": avg_time,
                    "time_str": minutes_to_hms_str(avg_time)
                }

    # ---------- Image 1: GENERAL STATS (single table) ----------
    # Weeks for averages
    weekly = df.copy()
    weekly["Week"] = weekly["Date"].dt.to_period(
        "W").apply(lambda r: r.start_time)
    n_weeks = max(1, weekly["Week"].nunique())

    total_distance = df["Distance"].sum()
    total_time_min = df["Duration (min)"].sum()
    n_activities = len(df)
    avg_dist_per_week = total_distance / n_weeks
    avg_time_per_week = total_time_min / n_weeks
    overall_avg_pace = df["Avg Pace (min/km)"].mean()
    overall_avg_hr = df["Avg HR"].mean()
    longest_run = df["Distance"].max()
    fastest_pace = df["Avg Pace (min/km)"].min()

    general_stats = pd.DataFrame([
        ["Total Distance (km)", round(total_distance, 1)],
        ["Total Time (hh:mm:ss)", minutes_to_hms_str(total_time_min)],
        ["Activities", n_activities],
        ["Weeks", n_weeks],
        ["Avg Distance / Week (km)", round(avg_dist_per_week, 1)],
        ["Avg Time / Week (hh:mm:ss)", minutes_to_hms_str(avg_time_per_week)],
        ["Overall Avg Pace (min/km)", min_per_km_to_str(overall_avg_pace)],
        ["Overall Avg HR (bpm)", int(round(overall_avg_hr, 0))],
        ["Longest Run (km)", round(longest_run, 1)],
        ["Fastest Avg Pace (min/km)", min_per_km_to_str(fastest_pace)],
    ], columns=["Metric", "Value"])

    export_table_styled(
        general_stats,
        filename=os.path.join(OUTPUT_DIR, "general_stats.png"),
        caption="General Stats (Overall & Per-Week Averages)"
    )

    # ---------- Image 2: PREDICTION CHARTS ----------
    plot_predictions(
        df=df,
        long_runs=long_runs,
        weekly_longest=weekly_longest,
        pred_all=pred_all,
        pred_long=pred_long,
        pred_week=pred_week,
        blended=blended,
        out_path=os.path.join(OUTPUT_DIR, "race_predictions.png")
    )

    # ---------- Image 3: WEEKLY TRENDS ----------
    plot_weekly_trends(
        df=df,
        out_path=os.path.join(OUTPUT_DIR, "weekly_trends.png")
    )

    # ---------- Image 4: PREDICTIONS TABLE (pretty) ----------
    preds_rows = []

    # Create rows for each race distance
    for race_name in RACE_DISTANCES.keys():
        for p in [pred_all, pred_long, pred_week]:
            if p.get("ok") and p.get(race_name):
                race_data = p[race_name]
                preds_rows.append([
                    race_name,
                    p["label"],
                    f"{min_per_km_to_str(race_data['pace_est'])} ({race_data['time_str']})"
                ])
            else:
                preds_rows.append([race_name, p["label"], "(no prediction)"])

        # Add blended prediction for this race
        if blended and blended.get(race_name):
            race_data = blended[race_name]
            preds_rows.append([
                race_name,
                "Blended Average",
                f"{min_per_km_to_str(race_data['pace_est'])} ({race_data['time_str']})"
            ])

    preds_df = pd.DataFrame(preds_rows, columns=[
        "Race Distance",
        "Prediction Model",
        "Pace & Time"
    ])

    export_table_styled(
        preds_df,
        filename=os.path.join(OUTPUT_DIR, "race_predictions_table.png"),
        caption="Race Predictions (Riegel, exponent=1.06)"
    )

    # Console confirmation
    print("\nSaved files in './output':")
    print(" - general_stats.png")
    print(" - race_predictions.png")
    print(" - weekly_trends.png")
    print(" - race_predictions_table.png")
    if not _HAVE_DFI:
        print("\n(Install 'dataframe_image' for nicer table rendering: pip install dataframe_image)")


if __name__ == "__main__":
    main()
