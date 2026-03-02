#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rematch_profiler.py
===================
订阅 /rematch_statistics，实时显示 rematch 节点各阶段耗时统计，
并在退出时打印汇总表格，可选保存 CSV 以供离线分析。

用法:
    rosrun umap rematch_profiler.py
    rosrun umap rematch_profiler.py _csv_path:=/tmp/rematch_stats.csv
    rosrun umap rematch_profiler.py _window:=100 _plot:=true
"""

import os
import sys
import csv
import signal
import threading
import collections
from datetime import datetime

import rospy
from umap.msg import RematchStatistics

# ── 可选: matplotlib 实时绘图 ─────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")          # 若无 Tk 可改为 "Qt5Agg" 或 "Agg"
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── 阶段定义 ──────────────────────────────────────────────────────────────────
STAGES = [
    ("t_parse_ms",       "Parse"),
    ("t_cost_matrix_ms", "CostMatrix"),
    ("t_association_ms", "Munkres"),
    ("t_update_ms",      "KF Update"),
    ("t_publish_vis_ms", "PubVis"),
    ("t_publish_det_ms", "PubDetails"),
    ("t_total_ms",       "TOTAL"),
]
STAGE_FIELDS  = [s[0] for s in STAGES]
STAGE_LABELS  = [s[1] for s in STAGES]
WINDOW_COLOR  = "\033[1;36m"   # cyan
WARN_COLOR    = "\033[1;33m"   # yellow
RESET_COLOR   = "\033[0m"


class RematchProfiler:
    def __init__(self):
        rospy.init_node("rematch_profiler", anonymous=False)

        # Parameters
        self.window    = int(rospy.get_param("~window",   200))    # rolling window size
        self.csv_path  = rospy.get_param("~csv_path",    "")
        self.plot      = rospy.get_param("~plot",        False) and HAS_MPL
        self.warn_ms   = float(rospy.get_param("~warn_total_ms", 100.0))

        # Rolling buffers (one deque per field)
        self._lock   = threading.Lock()
        self._bufs   = {f: collections.deque(maxlen=self.window) for f in STAGE_FIELDS}
        self._counts = {f: 0 for f in STAGE_FIELDS}   # total sample count
        self._sums   = {f: 0.0 for f in STAGE_FIELDS}
        self._maxes  = {f: 0.0 for f in STAGE_FIELDS}
        self._n_msgs = 0

        # CSV writer
        self._csv_file   = None
        self._csv_writer = None
        if self.csv_path:
            try:
                self._csv_file   = open(self.csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=["timestamp", "num_det", "num_trk",
                                "num_matched", "num_unmatched"] + STAGE_FIELDS
                )
                self._csv_writer.writeheader()
                rospy.loginfo("[Profiler] CSV output -> %s", self.csv_path)
            except IOError as e:
                rospy.logwarn("[Profiler] Cannot open CSV: %s", e)

        # ROS subscriber
        self._sub = rospy.Subscriber(
            "/rematch_statistics", RematchStatistics,
            self._callback, queue_size=50
        )

        # Matplotlib setup
        if self.plot:
            self._setup_plot()

        # Shutdown hook
        rospy.on_shutdown(self._on_shutdown)
        signal.signal(signal.SIGINT, self._sigint)

        rospy.loginfo("[Profiler] Listening on /rematch_statistics  "
                      "(window=%d, warn_ms=%.0f)", self.window, self.warn_ms)

    # ── ROS callback ──────────────────────────────────────────────────────────
    def _callback(self, msg: RematchStatistics):
        with self._lock:
            self._n_msgs += 1
            for field in STAGE_FIELDS:
                val = getattr(msg, field)
                self._bufs[field].append(val)
                self._sums[field]  += val
                self._counts[field] += 1
                if val > self._maxes[field]:
                    self._maxes[field] = val

            # CSV row
            if self._csv_writer:
                row = {
                    "timestamp":    msg.header.stamp.to_sec(),
                    "num_det":      msg.num_detections,
                    "num_trk":      msg.num_trackers,
                    "num_matched":  msg.num_matched,
                    "num_unmatched": msg.num_unmatched_det,
                }
                for f in STAGE_FIELDS:
                    row[f] = getattr(msg, f)
                self._csv_writer.writerow(row)

        # Warn if total latency is too high
        if msg.t_total_ms > self.warn_ms:
            rospy.logwarn_throttle(
                2.0,
                "[Profiler] HIGH LATENCY: total=%.1f ms  "
                "(det=%d trk=%d match=%d)",
                msg.t_total_ms,
                msg.num_detections, msg.num_trackers, msg.num_matched
            )

        # Console print every 10 messages
        if self._n_msgs % 10 == 0:
            self._print_live(msg)

    # ── Console output ────────────────────────────────────────────────────────
    def _print_live(self, last_msg: RematchStatistics):
        with self._lock:
            lines = [
                f"\n{WINDOW_COLOR}══════════════════════════════════════════════════{RESET_COLOR}",
                f"  [Rematch Profiler]  frame #{self._n_msgs}   "
                f"det={last_msg.num_detections}  trk={last_msg.num_trackers}  "
                f"matched={last_msg.num_matched}  new={last_msg.num_unmatched_det}",
                f"  {'Stage':<14}  {'Last(ms)':>9}  {'Mean(ms)':>9}  {'Max(ms)':>9}  {'% of total':>10}",
                f"  {'-'*14}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}",
            ]
            total_mean = self._sums["t_total_ms"] / max(self._counts["t_total_ms"], 1)
            for field, label in STAGES:
                last_v = list(self._bufs[field])[-1] if self._bufs[field] else 0.0
                mean_v = self._sums[field] / max(self._counts[field], 1)
                max_v  = self._maxes[field]
                pct    = 100.0 * mean_v / max(total_mean, 1e-6) if field != "t_total_ms" else 100.0
                tag    = f"{WARN_COLOR}◀ SLOW{RESET_COLOR}" if (field == "t_total_ms" and last_v > self.warn_ms) else ""
                lines.append(
                    f"  {label:<14}  {last_v:>9.2f}  {mean_v:>9.2f}  {max_v:>9.2f}  {pct:>9.1f}%  {tag}"
                )
            lines.append(f"{WINDOW_COLOR}══════════════════════════════════════════════════{RESET_COLOR}\n")
        print("\n".join(lines), flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    def _print_summary(self):
        if self._n_msgs == 0:
            print("[Profiler] No messages received.")
            return

        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  REMATCH PROFILER SUMMARY  ({self._n_msgs} frames)")
        print(sep)
        print(f"  {'Stage':<14}  {'Mean(ms)':>9}  {'Std(ms)':>9}  {'Max(ms)':>9}  {'% total':>8}")
        print(f"  {'-'*14}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")

        total_mean = self._sums["t_total_ms"] / max(self._counts["t_total_ms"], 1)
        for field, label in STAGES:
            n      = self._counts[field]
            mean_v = self._sums[field] / max(n, 1)
            max_v  = self._maxes[field]
            # std from rolling buffer (approximate over last `window` samples)
            buf    = list(self._bufs[field])
            if len(buf) > 1:
                import statistics
                std_v = statistics.stdev(buf)
            else:
                std_v = 0.0
            pct = 100.0 * mean_v / max(total_mean, 1e-6) if field != "t_total_ms" else 100.0
            print(f"  {label:<14}  {mean_v:>9.2f}  {std_v:>9.2f}  {max_v:>9.2f}  {pct:>7.1f}%")

        print(sep)
        if self.csv_path and self._csv_file:
            print(f"  CSV saved to: {self.csv_path}")
        print(sep + "\n")

    # ── Matplotlib live plot ──────────────────────────────────────────────────
    def _setup_plot(self):
        self._fig, self._axes = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={"height_ratios": [3, 1]}
        )
        self._fig.suptitle("Rematch Node — Per-stage Latency", fontsize=13)
        self._ani = animation.FuncAnimation(
            self._fig, self._update_plot, interval=500, cache_frame_data=False
        )

    def _update_plot(self, _frame):
        with self._lock:
            ax1, ax2 = self._axes
            ax1.cla()
            ax2.cla()

            # ── Time-series: each stage except total ──────────────────────────
            for field, label in STAGES[:-1]:   # skip TOTAL
                buf = list(self._bufs[field])
                if buf:
                    ax1.plot(buf, label=label, linewidth=1.2)
            ax1.set_ylabel("ms")
            ax1.set_title("Rolling latency per stage (last %d frames)" % self.window)
            ax1.legend(loc="upper right", fontsize=8)
            ax1.grid(True, alpha=0.3)

            # ── Bar: mean breakdown ───────────────────────────────────────────
            labels_bar = STAGE_LABELS[:-1]     # skip TOTAL
            means_bar  = [self._sums[f] / max(self._counts[f], 1)
                          for f in STAGE_FIELDS[:-1]]
            bars = ax2.bar(labels_bar, means_bar)
            ax2.set_ylabel("Mean (ms)")
            ax2.set_title("Mean latency breakdown")
            ax2.grid(True, alpha=0.3, axis="y")
            for bar, val in zip(bars, means_bar):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.1,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        self._fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def _on_shutdown(self):
        self._print_summary()
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
                rospy.loginfo("[Profiler] CSV closed.")
            except Exception:
                pass

    def _sigint(self, sig, frame):
        rospy.signal_shutdown("SIGINT")
        sys.exit(0)

    def spin(self):
        if self.plot:
            plt.show(block=False)
        rospy.spin()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    profiler = RematchProfiler()
    profiler.spin()
