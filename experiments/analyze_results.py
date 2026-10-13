#!/usr/bin/env python3
"""
analyze_results.py — round1/round2の検出ログを座標クラスタリングで11ドアに
統合し、正解ラベル(expected_free_width)と突き合わせてレポートを出す。

前提とするファイル (~/lidar_ws/experiments/logs/ に置く想定):
    round1_clean.csv   : door_detector_node のCSVログ(round1区間のみ)
    round2_clean.csv   : door_detector_node のCSVログ(round2区間のみ)

door_detector_node のCSV列:
    stamp, robot_x, robot_y, door_id, door_x, door_y, dist_to_door,
    source, state, confirmed, hit_count,
    horizontal_gap, passable_confirmed, confirmed_free_width

使い方:
    cd ~/lidar_ws/experiments
    python3 analyze_results.py

出力:
    logs/round1_report.csv, logs/round2_report.csv (ドアごとの集計)
    logs/round1_scatter.png, logs/round2_scatter.png (正解 vs 検出の散布図)
    logs/round1_timeseries.csv, logs/round2_timeseries.csv (ドアごとの時系列生データ)
    logs/round1_timeseries.png, logs/round2_timeseries.png (ドアごとの時系列グラフ、10ドア分をグリッド表示)
    logs/combined_timeseries.csv (round1/round2を1本にまとめた時系列生データ、round列で区別)
    logs/combined_timeseries.png (round1/round2を同じ時間軸上に重ねたドアごとのグラフ)
    logs/robot_trajectory.csv (round1/round2のロボット位置、round列で区別)
    logs/robot_trajectory.png (走行軌跡(XY平面)とX座標の時系列、round1+round2統合)
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

# グラフ内に日本語を書くため、入ってる日本語フォントを自動で探して設定する。
# 見つからなければ警告だけ出して続行する(文字化け=豆腐になる可能性あり)。
_JP_FONT_CANDIDATES = [
    "Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic",
    "VL Gothic", "Yu Gothic", "MS Gothic", "Hiragino Sans",
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _name in _JP_FONT_CANDIDATES:
    if _name in _available_fonts:
        matplotlib.rcParams["font.family"] = _name
        break
else:
    print("[warn] 日本語フォントが見つかりませんでした。文字化けする場合は "
          "'sudo apt install fonts-noto-cjk' 等でフォントを追加してください。")
matplotlib.rcParams["axes.unicode_minus"] = False

LOG_DIR = Path.home() / "lidar_ws" / "experiments" / "logs"

# corridor10_world.sdf の <model><pose> から算出した、各ドアの
# 「開口部の中心座標」(map座標系、ワールド原点基準)。
# hinge位置 + 0.5 * (frame_widthの方向ベクトル) で算出。
# entrance_door は面一設置のため recess/gap いずれの方式でも検出不可能と
# 判断し、集計対象からは除外する(西側入口の交差点エリアはノイズの温床でも
# あるため、ここに含めると誤検出が entrance_door の実績として混入してしまう)
DOOR_CENTERS = {
    "door_north_1": (1.5, 1.075),
    "door_south_1": (1.5, -1.075),
    "door_north_2": (4.5, 1.075),
    "door_south_2": (4.5, -1.075),
    "door_north_3": (7.5, 1.075),
    "door_south_3": (7.5, -1.075),
    "door_north_4": (10.5, 1.075),
    "door_south_4": (10.5, -1.075),
    "door_north_5": (13.5, 1.075),
    "door_south_5": (13.5, -1.075),
}

# 各ラウンドの正解角度パターン (sim_timeでの投入順ではなく、最終的に
# 反映された値をそのまま記載。door_ctrl.py の ground_truth ログと
# 揃えたい場合はそちらの数値を書き換えて使ってください)
PATTERN_A = {
    "door_north_1": 0.0,
    "door_south_1": 0.0,
    "door_north_2": 1.5708,
    "door_south_2": 0.5,
    "door_north_3": 0.0,
    "door_south_3": 1.5708,
    "door_north_4": 1.3,
    "door_south_4": 0.0,
    "door_north_5": 0.9,
    "door_south_5": 0.0,
}

PATTERN_B = {
    "door_north_1": 0.8,
    "door_south_1": 0.0,
    "door_north_2": 0.4,
    "door_south_2": 0.0,
    "door_north_3": 1.3,
    "door_south_3": 1.5708,
    "door_north_4": 0.0,
    "door_south_4": 1.5708,
    "door_north_5": 0.0,
    "door_south_5": 0.7,
}

FRAME_WIDTH = 1.0
LEAF_LENGTH = 0.95


def expected_free_width(angle_rad: float) -> float:
    return max(0.0, FRAME_WIDTH - LEAF_LENGTH * math.cos(angle_rad))


def expected_state(angle_rad: float, thresh_rad: float = 0.15) -> str:
    """leaf_angle_thresh(0.15rad)を基準にした、大まかな正解open/closed判定。
    厳密なbeyond_open判定の再現ではなく、目安として使う。"""
    return "open" if angle_rad > thresh_rad else "closed"


def nearest_door(x: float, y: float):
    best_name, best_dist = None, float("inf")
    for name, (dx, dy) in DOOR_CENTERS.items():
        d = math.hypot(x - dx, y - dy)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name, best_dist


def load_and_assign(csv_path: Path, max_assign_dist: float = 1.0):
    """CSVを読み、各行のdoor_x,door_yから最寄りの物理ドア名を割り当てる。
    max_assign_distより遠い場合はノイズ(None)として除外する。"""
    rows_by_door = defaultdict(list)
    n_total, n_dropped = 0, 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            try:
                stamp = float(row["stamp"])
                dx = float(row["door_x"])
                dy = float(row["door_y"])
                gap = float(row["horizontal_gap"])
                state = row["state"]
                dist_to_door = float(row["dist_to_door"])
            except (KeyError, ValueError):
                n_dropped += 1
                continue
            name, dist = nearest_door(dx, dy)
            if dist > max_assign_dist:
                n_dropped += 1
                continue
            rows_by_door[name].append({
                "stamp": stamp, "gap": gap, "state": state, "dist_to_door": dist_to_door,
            })
    return rows_by_door, n_total, n_dropped


# ロボットがdoor_detector_node側のpassability_confirm_dist(デフォルト1.0m)
# まで近づいた時だけ、horizontal_gapの信頼性が上がるという設計上の想定を
# 検証するための、近距離限定の平均値指標。
NEAR_DIST_THRESH = 1.0


def summarize(rows_by_door, pattern: dict, out_csv: Path):
    fieldnames = [
        "door_name", "n_samples", "expected_angle", "expected_free_width",
        "expected_state", "detected_avg_gap", "detected_open_ratio",
        "gap_error", "n_gap_known",
        "near_n_samples", "near_avg_gap", "near_gap_error",
    ]
    lines = []
    for name, angle in pattern.items():
        exp_gap = expected_free_width(angle)
        exp_state = expected_state(angle)
        samples = rows_by_door.get(name, [])
        # gap>=0(既知)のサンプルだけで平均を取る。gap<0は「closed/判定不能」の印
        known = [s for s in samples if s["gap"] >= 0]
        near_samples = [s for s in known if s["dist_to_door"] <= NEAR_DIST_THRESH]
        if samples:
            open_ratio = sum(1 for s in samples if s["state"] == "open") / len(samples)
        else:
            open_ratio = float("nan")
        if known:
            avg_gap = sum(s["gap"] for s in known) / len(known)
        else:
            avg_gap = float("nan")
        if near_samples:
            near_avg_gap = sum(s["gap"] for s in near_samples) / len(near_samples)
        else:
            near_avg_gap = float("nan")
        lines.append({
            "door_name": name,
            "n_samples": len(samples),
            "expected_angle": angle,
            "expected_free_width": round(exp_gap, 3),
            "expected_state": exp_state,
            "detected_avg_gap": round(avg_gap, 3) if known else "",
            "detected_open_ratio": round(open_ratio, 3) if samples else "",
            "gap_error": round(avg_gap - exp_gap, 3) if known else "",
            "n_gap_known": len(known),
            "near_n_samples": len(near_samples),
            "near_avg_gap": round(near_avg_gap, 3) if near_samples else "",
            "near_gap_error": round(near_avg_gap - exp_gap, 3) if near_samples else "",
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)
    return lines


def plot_scatter(lines, out_png: Path, title: str):
    # ドアごとに固有の色を割り当てる(tab10は10色あるので10ドアにちょうど対応)
    door_names = [line["door_name"] for line in lines]
    cmap = plt.get_cmap("tab10")
    color_map = {name: cmap(i % 10) for i, name in enumerate(door_names)}

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", zorder=1, linewidth=1)
    ax.axvline(0.7, color="red", linestyle=":", linewidth=1, zorder=1)

    points = []  # (x, y_all, name, color)
    for line in lines:
        if line["detected_avg_gap"] == "":
            continue
        name = line["door_name"]
        color = color_map[name]
        x = line["expected_free_width"]
        y_all = line["detected_avg_gap"]
        ax.scatter([x], [y_all], color=[color], marker="o", s=60, zorder=3)
        points.append((x, y_all, name, color))

        if line["near_avg_gap"] != "":
            y_near = line["near_avg_gap"]
            ax.scatter([x], [y_near], color=[color], marker="^", s=60, zorder=3)
            # all avg -> near avg の変化を細い矢印でつなぐ(同じドアだと分かるように)
            ax.annotate(
                "", xy=(x, y_near), xytext=(x, y_all),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.6),
                zorder=2,
            )

    # 近接する点(特に原点付近に固まる全閉ドア群)は個別ラベルを付けず、
    # どの点がどのドアかを右側凡例の下に注記としてまとめる
    EPS = 0.04
    used = [False] * len(points)
    note_lines = []
    for i in range(len(points)):
        if used[i]:
            continue
        xi, yi, ni, _ = points[i]
        group = [ni]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            xj, yj, nj, _ = points[j]
            if abs(xi - xj) < EPS and abs(yi - yj) < EPS:
                group.append(nj)
                used[j] = True
        if len(group) > 1 and xi < 0.15 and yi < 0.15:
            note_lines.append(f"({xi:.2f}, {yi:.2f}): " + ", ".join(group))

    ax.set_xlabel("正解の通行幅 expected_free_width [m]", fontsize=15)
    ax.set_ylabel("検出した平均通行幅 detected_avg_gap [m]", fontsize=15)
    ax.set_title(title, fontsize=17)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(labelsize=12)

    # マーカー形状の意味を示す小さい凡例(左上、プロット内)
    shape_legend = [
        Line2D([0], [0], color="gray", linestyle="--", label="理想線 (y=x)"),
        Line2D([0], [0], color="red", linestyle=":", label="通行可能しきい値 (0.7m)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=9, label="全サンプル平均"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=9, label=f"近距離(\u2264{NEAR_DIST_THRESH:.1f}m)平均"),
    ]
    shape_leg = ax.legend(handles=shape_legend, fontsize=12, loc="upper left",
                           framealpha=0.95, edgecolor="black", title="マーカーの意味",
                           title_fontsize=13)
    ax.add_artist(shape_leg)

    # ドアごとの色の凡例は右側にまとめる
    door_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[name],
               markersize=9, label=name)
        for name in door_names
    ]
    ax.legend(handles=door_handles, fontsize=12, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0, title="ドア",
              title_fontsize=13, framealpha=0.95)

    # 密集している点の内訳を、図の下部に注記として書く
    if note_lines:
        note_text = "重なっている点  " + " / ".join(note_lines)
        fig.text(0.5, 0.01, note_text, ha="center", va="bottom", fontsize=12,
                  bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_timeseries(rows_by_door, pattern: dict, out_csv: Path):
    """ドアごとの時系列生データ(stamp, horizontal_gap, state)をロング形式で書き出す。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["door_name", "stamp", "horizontal_gap", "state", "expected_free_width"])
        for name, angle in pattern.items():
            exp_gap = round(expected_free_width(angle), 3)
            samples = sorted(rows_by_door.get(name, []), key=lambda s: s["stamp"])
            for s in samples:
                writer.writerow([name, s["stamp"], s["gap"], s["state"], exp_gap])


def plot_timeseries(rows_by_door, pattern: dict, out_png: Path, title: str):
    """ドアごとにhorizontal_gapの時系列を、10ドア分まとめてグリッド表示する。
    点の色はstate(open=緑, closed=赤)。gap>=0(既知)は丸、gap<0(closed/判定不能)は
    ×印でy=0に描く。破線は正解のexpected_free_width。"""
    names = list(pattern.keys())
    n = len(names)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)

    all_stamps = [s["stamp"] for samples in rows_by_door.values() for s in samples]
    if all_stamps:
        xmin, xmax = min(all_stamps), max(all_stamps)
        margin = (xmax - xmin) * 0.02 if xmax > xmin else 1.0
        xmin, xmax = xmin - margin, xmax + margin
    else:
        xmin, xmax = 0.0, 1.0

    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        samples = sorted(rows_by_door.get(name, []), key=lambda s: s["stamp"])
        if samples:
            known = [s for s in samples if s["gap"] >= 0]
            unknown = [s for s in samples if s["gap"] < 0]
            if known:
                ax.scatter(
                    [s["stamp"] for s in known], [s["gap"] for s in known],
                    c=["tab:green" if s["state"] == "open" else "tab:red" for s in known],
                    marker="o", s=8, zorder=3,
                )
            if unknown:
                ax.scatter(
                    [s["stamp"] for s in unknown], [0.0] * len(unknown),
                    c=["tab:green" if s["state"] == "open" else "tab:red" for s in unknown],
                    marker="x", s=14, zorder=3,
                )
        exp_gap = expected_free_width(pattern[name])
        ax.axhline(exp_gap, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.7, color="black", linestyle=":", linewidth=0.7)
        ax.set_title(name, fontsize=13)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("経過時間 stamp [s]", fontsize=12)
        ax.set_ylabel("通行幅 horizontal_gap [m]", fontsize=12)
        ax.tick_params(labelsize=11)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=18)
    fig.text(0.5, 0.005,
              "（緑=open、赤=closed、○=通行幅あり、×=不明/判定なし、グレー破線=正解値、黒点線=0.7mしきい値）",
              ha="center", fontsize=12)
    fig.tight_layout(rect=(0.01, 0.03, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_combined_timeseries(round_data: dict, out_png: Path, title: str):
    """round1/round2両方を、ドアごとに1枚のグラフ内で表示する。
    マーカー形状でround1(丸)/round2(四角)を区別し、色はstate(open=緑,closed=赤)。
    正解値(expected_free_width)はラウンドごとに変わるので、その時間区間だけ
    横線を引く。ラウンド間の無データ区間(ドア状態を切り替えてた待機時間)は
    詰めて表示し、round1の開始を時刻0とする(実時間の対応は失われるが、
    プロットが間延びしないメリットがある)。"""
    first_pattern = next(iter(round_data.values()))[1]
    names = list(first_pattern.keys())
    n = len(names)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)

    round_markers = {"round1": "o", "round2": "s"}

    # ラウンドごとの実際のstamp範囲を求め、ラウンド間の空白を詰めた
    # 「圧縮時間軸」への変換オフセットを計算する(round1開始を0秒とする)。
    # ラウンド同士は隙間なく直接つなげる(GAP=0)。
    GAP = 0.0
    round_ranges = {}
    for round_name, (rows_by_door, _) in round_data.items():
        stamps = [s["stamp"] for samples in rows_by_door.values() for s in samples]
        round_ranges[round_name] = (min(stamps), max(stamps)) if stamps else (0.0, 0.0)

    offsets = {}
    cursor = 0.0
    for round_name in round_data:  # dict順(round1, round2の順を想定)
        rmin, rmax = round_ranges[round_name]
        offsets[round_name] = cursor - rmin
        cursor += (rmax - rmin) + GAP
    xmax_total = cursor - GAP

    def compressed(round_name, stamp):
        return stamp + offsets[round_name]

    def find_visits(samples, gap_threshold=5.0):
        """時刻でソート済みのsamplesを、stamp間隔がgap_thresholdを超える箇所で
        「訪問(通過)」ごとに分割する。往復走行なら通常1ラウンドにつき2回
        (行き/帰り)ドア前を通過するので、visitも2つに分かれるはず。"""
        if not samples:
            return []
        visits = [[samples[0]]]
        for s in samples[1:]:
            if s["stamp"] - visits[-1][-1]["stamp"] > gap_threshold:
                visits.append([s])
            else:
                visits[-1].append(s)
        return visits

    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        closest_points = []  # (cx, dist_to_door) を全ラウンド分ここに集める
        for round_name, (rows_by_door, pattern) in round_data.items():
            samples = sorted(rows_by_door.get(name, []), key=lambda s: s["stamp"])
            if not samples:
                continue
            marker = round_markers.get(round_name, "o")
            known = [s for s in samples if s["gap"] >= 0]
            unknown = [s for s in samples if s["gap"] < 0]
            if known:
                ax.scatter(
                    [compressed(round_name, s["stamp"]) for s in known],
                    [s["gap"] for s in known],
                    c=["tab:green" if s["state"] == "open" else "tab:red" for s in known],
                    marker=marker, s=10, zorder=3,
                )
            if unknown:
                # gap<0(closed/判定不能)はy=0に×印で描く
                ax.scatter(
                    [compressed(round_name, s["stamp"]) for s in unknown],
                    [0.0] * len(unknown),
                    c=["tab:green" if s["state"] == "open" else "tab:red" for s in unknown],
                    marker="x", s=14, zorder=3,
                )
            stamps = [compressed(round_name, s["stamp"]) for s in samples]
            exp_gap = expected_free_width(pattern[name])
            ax.hlines(exp_gap, min(stamps), max(stamps), color="gray",
                      linestyle="--", linewidth=1, zorder=2)

            # 通過(訪問)ごとに、ロボットがドアに最も近づいた時刻を縦線で示す
            for visit in find_visits(samples):
                closest = min(visit, key=lambda s: s.get("dist_to_door", float("inf")))
                cx = compressed(round_name, closest["stamp"])
                ax.axvline(cx, color="goldenrod", linewidth=1.3, alpha=0.7, zorder=2.5)
                closest_points.append((cx, closest["dist_to_door"]))

        # ラベルは全部同じ高さ(1段)に置く。近すぎて重なりそうなものは
        # クラスタとしてまとめ、そのクラスタの中心を軸に左右対称に振り分ける
        # (今までは先に来た方を基準に後の方だけ右へ押してたので非対称やった)。
        # 線(axvline)自体は必ずcxの位置に描くので、ラベルの位置だけが
        # 実際の時刻からわずかにずれる場合がある。
        closest_points.sort(key=lambda p: p[0])
        MIN_LABEL_SEP = max(xmax_total * 0.075, 1e-6)
        label_trans = blended_transform_factory(ax.transData, ax.transAxes)

        clusters = []
        for cx, d in closest_points:
            if clusters and cx - clusters[-1][-1][0] < MIN_LABEL_SEP:
                clusters[-1].append((cx, d))
            else:
                clusters.append([(cx, d)])

        for cluster in clusters:
            k = len(cluster)
            center = sum(c[0] for c in cluster) / k
            for idx, (cx, d) in enumerate(cluster):
                anchor = center + (idx - (k - 1) / 2) * MIN_LABEL_SEP
                ax.text(anchor, 1.03, f"{d:.2f}m", fontsize=6, ha="center", va="bottom",
                        color="goldenrod", zorder=6, transform=label_trans, clip_on=False,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.6))

        ax.axhline(0.7, color="black", linestyle=":", linewidth=0.7, zorder=1)
        # タイトルはラベル(1段固定)より必ず上に来る、axes基準の固定位置に描く
        ax.text(0.5, 1.20, name, transform=ax.transAxes, fontsize=13,
                fontweight="bold", ha="center", va="bottom", clip_on=False)
        ax.set_xlim(-xmax_total * 0.02, xmax_total * 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("経過時間 stamp [s]", fontsize=12)
        ax.set_ylabel("通行幅 horizontal_gap [m]", fontsize=12)
        ax.tick_params(labelsize=11)

        # ラウンドの切れ目を薄い縦線で示す(隙間はゼロ、目印だけ)
        round_names_in_order = list(round_data.keys())
        for k in range(len(round_names_in_order) - 1):
            r_end = round_ranges[round_names_in_order[k]][1]
            r_end_c = compressed(round_names_in_order[k], r_end)
            ax.axvline(r_end_c, color="gray", linewidth=0.8, alpha=0.6, zorder=0)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=18)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="round1"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="round2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green", markersize=8, label="open"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=8, label="closed"),
        Line2D([0], [0], color="gray", linestyle="--", label="正解値(ラウンドごと)"),
        Line2D([0], [0], color="black", linestyle=":", label="0.7mしきい値"),
        Line2D([0], [0], color="goldenrod", linewidth=1.5, label="最接近時刻(通過ごと、距離[m])"),
        Line2D([0], [0], marker="x", color="gray", linestyle="None", markersize=7, label="×=不明/判定なし"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.01, 0.07, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def load_robot_trajectory(csv_path: Path):
    """CSVからrobot_x, robot_yをstampごとに(重複除去して)読み込み、
    時刻順のリスト[(stamp, robot_x, robot_y), ...]を返す。
    同じstampの行は複数ドア分重複してるので、辞書で一度だけ拾う。"""
    seen = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stamp = float(row["stamp"])
                rx = float(row["robot_x"])
                ry = float(row["robot_y"])
            except (KeyError, ValueError):
                continue
            seen[stamp] = (rx, ry)
    return [(s, x, y) for s, (x, y) in sorted(seen.items())]


def moving_average(values, window=31):
    """単純な移動平均。teleop操作のジッターで軌跡がガタつくのを見やすく均す。
    生データそのものは変えず、表示用にだけ使う。"""
    n = len(values)
    if n == 0 or window <= 1:
        return list(values)
    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def despike(values, window=15, threshold=0.08):
    """周辺の値の中央値やなく、局所的な直線フィットからの外れを見て
    段差(ログの抜け前後でロボット位置が一瞬飛ぶ現象)を補正する。
    走行中は位置が右肩上がり/下がりに動き続けてるので、単純な中央値
    比較やと傾きに埋もれて段差を拾えない。直線フィットとの差分を見る
    ことで、緩やかな坂(=本物の動き)は残しつつ段差だけ検出できる。"""
    n = len(values)
    if n == 0 or window <= 2:
        return list(values)
    half = window // 2
    out = list(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xs = list(range(lo, hi))
        ys = values[lo:hi]
        m = len(xs)
        mean_x = sum(xs) / m
        mean_y = sum(ys) / m
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den else 0.0
        intercept = mean_y - slope * mean_x
        predicted = slope * i + intercept
        if abs(values[i] - predicted) > threshold:
            out[i] = predicted
    return out


def fix_gap_jumps(traj, gap_factor=4.0, settle=3):
    """(stamp,x,y)のリストから、stampの間隔が異常に広い箇所(=ログが
    欠けてた区間、door_detector_nodeがconfirmedゼロの間ロボット位置ごと
    記録してなかった箇所)を直接検出し、その直後の数点を「まだ本来の
    軌道に乗り切ってへん値」とみなして、ギャップ前の点とsettle点先の点を
    直線で結んで置き換える。振幅のしきい値で推測するdespikeと違って、
    実際に分かってるログの欠損箇所そのものを根拠にするので確実。"""
    n = len(traj)
    if n < 3:
        return [p[1] for p in traj], [p[2] for p in traj]
    stamps = [p[0] for p in traj]
    xs = [p[1] for p in traj]
    ys = [p[2] for p in traj]
    dts = [stamps[i + 1] - stamps[i] for i in range(n - 1)]
    sorted_dts = sorted(dts)
    med_dt = sorted_dts[len(sorted_dts) // 2] if sorted_dts else 0.0
    xs2, ys2 = list(xs), list(ys)
    if med_dt <= 0:
        return xs2, ys2
    i = 0
    while i < n - 1:
        if dts[i] > gap_factor * med_dt:
            j = min(i + 1 + settle, n - 1)
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[j], ys[j]
            span = j - i
            for k in range(1, span):
                t = k / span
                idx = i + k
                xs2[idx] = x0 + (x1 - x0) * t
                ys2[idx] = y0 + (y1 - y0) * t
            i = j
        else:
            i += 1
    return xs2, ys2


def smooth_path(values, window=3):
    """まず段差(飛び値)だけ直線フィットベースで補正してから、ごく軽い
    移動平均で仕上げる。窓を小さくすることで、折り返し・停止地点などの
    本物の角があまり丸まらないようにしている。"""
    despiked = despike(values)
    return moving_average(despiked, window=window)


def plot_robot_trajectory(traj_data: dict, out_png: Path, title: str):
    """round1/round2両方のロボット位置を、combined_timeseriesと同じ考え方
    (round1開始を0秒とし、ラウンド間の空白は詰めて直結)で2枚並べて表示する。
    左: 走行軌跡(XY平面、ドア位置も重ねる)
    右: X座標(進行方向)の時系列
    traj_data: {"round1": [(stamp,x,y),...], "round2": [...]}
    """
    round_markers = {"round1": "o", "round2": "s"}
    round_colors = {"round1": "tab:blue", "round2": "tab:purple"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- 左: 走行軌跡(XY平面) ---
    for round_name, traj in traj_data.items():
        if not traj:
            continue
        gx, gy = fix_gap_jumps(traj)
        xs = smooth_path(gx)
        ys = smooth_path(gy)
        ax1.plot(xs, ys, color=round_colors.get(round_name, "gray"),
                  marker=round_markers.get(round_name, "o"), markersize=1.5,
                  linewidth=1.8, alpha=0.85, label=round_name)

    for name, (dx, dy) in DOOR_CENTERS.items():
        ax1.scatter([dx], [dy], color="black", marker="x", s=45, zorder=5)
        ax1.annotate(name.replace("door_", ""), (dx, dy), fontsize=8,
                      xytext=(4, 4), textcoords="offset points")

    ax1.set_xlabel("robot_x [m]", fontsize=13)
    ax1.set_ylabel("robot_y [m]", fontsize=13)
    ax1.set_title("走行軌跡(XY平面、×はドア位置)", fontsize=14)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.tick_params(labelsize=11)
    ax1.legend(fontsize=10, loc="best")

    # --- 右: X座標(進行方向)の時系列。ラウンド間の空白は詰めて直結する ---
    GAP = 0.0
    ranges = {}
    for round_name, traj in traj_data.items():
        stamps = [p[0] for p in traj]
        if stamps:
            ranges[round_name] = (min(stamps), max(stamps))
    offsets = {}
    cursor = 0.0
    for round_name in traj_data:
        if round_name not in ranges:
            continue
        rmin, rmax = ranges[round_name]
        offsets[round_name] = cursor - rmin
        cursor += (rmax - rmin) + GAP
    xmax_total = cursor - GAP

    for round_name, traj in traj_data.items():
        if not traj or round_name not in offsets:
            continue
        ts = [p[0] + offsets[round_name] for p in traj]
        gx, _ = fix_gap_jumps(traj)
        xs = smooth_path(gx)
        ax2.plot(ts, xs, color=round_colors.get(round_name, "gray"),
                  marker=round_markers.get(round_name, "o"), markersize=1.5,
                  linewidth=1.8, alpha=0.85, label=round_name)

    for k, round_name in enumerate(list(traj_data.keys())[:-1]):
        if round_name in ranges:
            r_end_c = ranges[round_name][1] + offsets[round_name]
            ax2.axvline(r_end_c, color="gray", linewidth=0.8, alpha=0.6, zorder=0)

    # ドアのX位置を横線で示す(north/southで同じXのドアはまとめてラベルを付ける)。
    # ラベルのx座標はデータ範囲やなくaxes全体の割合(0〜1)で指定して、
    # 枠のサイズ(xlim)は変えずに、常に枠のすぐ外側に配置する。
    x_to_names = {}
    for name, (dx, _dy) in DOOR_CENTERS.items():
        x_to_names.setdefault(round(dx, 2), []).append(name.replace("door_", ""))
    door_label_trans = blended_transform_factory(ax2.transAxes, ax2.transData)
    for dx, names_here in sorted(x_to_names.items()):
        ax2.axhline(dx, color="black", linestyle=":", linewidth=0.7, alpha=0.5, zorder=0)
        ax2.text(1.02, dx, "/".join(names_here), fontsize=7,
                  va="center", ha="left", clip_on=False, transform=door_label_trans)

    ax2.set_xlabel("経過時間 stamp [s]", fontsize=13)
    ax2.set_ylabel("robot_x [m]", fontsize=13)
    ax2.set_title("走行位置(X座標)の時系列(統合、点線=ドアのX位置)", fontsize=14)
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=10, loc="upper left")

    fig.suptitle(title, fontsize=16)
    fig.text(0.5, 0.91, "※ 表示用に段差補正+軽い移動平均で平滑化(角はなるべく保持)",
              ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0.01, 0.02, 0.90, 0.88))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def export_robot_trajectory(traj_data: dict, out_csv: Path):
    """round1/round2のロボット位置をround列付きで1本のCSVにまとめる。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "stamp", "robot_x", "robot_y"])
        for round_name, traj in traj_data.items():
            for stamp, x, y in traj:
                writer.writerow([round_name, stamp, x, y])


def run_round(name: str, csv_name: str, pattern: dict):
    csv_path = LOG_DIR / csv_name
    if not csv_path.exists():
        print(f"[skip] {csv_path} が見つかりません")
        return None

    rows_by_door, n_total, n_dropped = load_and_assign(csv_path)

    print(f"\n=== {name} ===")
    print(f"元の行数: {n_total}, 除外(ノイズ/不明): {n_dropped}")
    for door_name in pattern:
        n = len(rows_by_door.get(door_name, []))
        print(f"  {door_name:16s}: {n:4d} サンプル")

    out_csv = LOG_DIR / f"{name}_report.csv"
    lines = summarize(rows_by_door, pattern, out_csv)
    print(f"レポート出力: {out_csv}")

    out_png = LOG_DIR / f"{name}_scatter.png"
    plot_scatter(lines, out_png, title=f"{name}: 正解 vs 検出 通行幅")
    print(f"散布図出力: {out_png}")

    ts_csv = LOG_DIR / f"{name}_timeseries.csv"
    export_timeseries(rows_by_door, pattern, ts_csv)
    print(f"時系列CSV出力: {ts_csv}")

    ts_png = LOG_DIR / f"{name}_timeseries.png"
    plot_timeseries(rows_by_door, pattern, ts_png, title=f"{name}: 通行幅の時系列(ドアごと)")
    print(f"時系列グラフ出力: {ts_png}")

    return rows_by_door


def export_combined_timeseries(round_data: dict, out_csv: Path):
    """round1/round2両方の時系列生データを、round列付きで1本のCSVにまとめる。
    round_data: {"round1": (rows_by_door, pattern), "round2": (rows_by_door, pattern)}
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "door_name", "stamp", "horizontal_gap", "state", "expected_free_width"])
        for round_name, (rows_by_door, pattern) in round_data.items():
            for name, angle in pattern.items():
                exp_gap = round(expected_free_width(angle), 3)
                samples = sorted(rows_by_door.get(name, []), key=lambda s: s["stamp"])
                for s in samples:
                    writer.writerow([round_name, name, s["stamp"], s["gap"], s["state"], exp_gap])


def main():
    round_data = {}

    rows1 = run_round("round1", "round1_clean.csv", PATTERN_A)
    if rows1 is not None:
        round_data["round1"] = (rows1, PATTERN_A)

    rows2 = run_round("round2", "round2_clean.csv", PATTERN_B)
    if rows2 is not None:
        round_data["round2"] = (rows2, PATTERN_B)

    if round_data:
        combined_csv = LOG_DIR / "combined_timeseries.csv"
        export_combined_timeseries(round_data, combined_csv)
        print(f"\n統合時系列CSV出力: {combined_csv}")

        combined_png = LOG_DIR / "combined_timeseries.png"
        plot_combined_timeseries(round_data, combined_png,
                                  title="round1+round2: 通行幅の時系列(統合)")
        print(f"統合時系列グラフ出力: {combined_png}")

    # ロボットの走行位置(軌跡・X座標の時系列)も同様にround1+round2を統合して出す
    traj_data = {}
    for round_name, csv_name in [("round1", "round1_clean.csv"), ("round2", "round2_clean.csv")]:
        csv_path = LOG_DIR / csv_name
        if csv_path.exists():
            traj_data[round_name] = load_robot_trajectory(csv_path)

    if traj_data:
        traj_csv = LOG_DIR / "robot_trajectory.csv"
        export_robot_trajectory(traj_data, traj_csv)
        print(f"ロボット軌跡CSV出力: {traj_csv}")

        traj_png = LOG_DIR / "robot_trajectory.png"
        plot_robot_trajectory(traj_data, traj_png,
                               title="round1+round2: ロボット走行位置(統合)")
        print(f"ロボット軌跡グラフ出力: {traj_png}")


if __name__ == "__main__":
    main()
