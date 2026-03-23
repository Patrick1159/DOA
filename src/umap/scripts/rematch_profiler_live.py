#!/home/user/miniconda3/envs/oa/bin/python3
"""
rematch_profiler_live.py
========================
实时订阅三个 rematch 统计话题，在终端中以固定频率刷新各节点的耗时分布，
并在 Ctrl-C 退出后输出汇总报告。

订阅的话题:
  /rematch_statistics_sam         (umap/RematchStatistics)
  /rematch_statistics_clip        (umap/RematchStatistics)
  /rematch_statistics_aggregator  (umap/RematchStatistics)

用法:
  rosrun umap rematch_profiler_live.py
  # 或直接运行（确保 ROS 环境已 source）
  python3 rematch_profiler_live.py
"""

import sys
import math
import signal
import threading
from collections import deque, defaultdict
from typing import Optional, Dict, List

import rospy
from umap.msg import RematchStatistics

# ─── 配置 ──────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 50          # 滑动窗口长度（帧数）
REFRESH_HZ    = 2.0         # 终端刷新频率
NODES         = ['sam', 'clip', 'aggregator']
TOPICS        = {
    'sam':         '/rematch_statistics_sam',
    'clip':        '/rematch_statistics_clip',
    'aggregator':  '/rematch_statistics_aggregator',
}

# ─── 颜色 ──────────────────────────────────────────────────────────────────
RESET  = '\033[0m'
BOLD   = '\033[1m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RED    = '\033[91m'
DIM    = '\033[2m'

# ─── 统计存储 ───────────────────────────────────────────────────────────────
class NodeStats:
    """滑动窗口 + 全局累计统计"""
    FIELDS = ['total_time', 'e2e_latency', 'model_time', 'preprocess_time',
              'postprocess_time', 'avg_model_time_per_obstacle',
              'sync_wait_time', 'obstacle_count']

    def __init__(self, window: int = WINDOW_SIZE):
        self._lock  = threading.Lock()
        self.window = window
        self.win: Dict[str, deque] = {f: deque(maxlen=window) for f in self.FIELDS}
        self.all: Dict[str, List]  = {f: []                   for f in self.FIELDS}
        self.recv_count = 0

    def update(self, msg: RematchStatistics):
        with self._lock:
            self.recv_count += 1
            for f in self.FIELDS:
                v = float(getattr(msg, f))
                self.win[f].append(v)
                self.all[f].append(v)

    def snapshot(self) -> dict:
        """返回当前滑动窗口统计（线程安全）"""
        with self._lock:
            result: Dict = {'recv_count': self.recv_count}
            for f in self.FIELDS:
                data = list(self.win[f])
                if not data:
                    result[f] = None
                    continue
                n    = len(data)
                mean = sum(data) / n
                mx   = max(data)
                mn   = min(data)
                # P95
                sorted_d = sorted(data)
                p95_idx  = min(int(math.ceil(0.95 * n)) - 1, n - 1)
                p95      = sorted_d[p95_idx]
                result[f] = dict(mean=mean, max=mx, min=mn, p95=p95, n=n)
        return result

    def summary(self) -> dict:
        """全程汇总统计"""
        with self._lock:
            result: Dict = {'recv_count': self.recv_count}
            for f in self.FIELDS:
                data = self.all[f]
                if not data:
                    result[f] = None
                    continue
                n    = len(data)
                mean = sum(data) / n
                mx   = max(data)
                mn   = min(data)
                sorted_d = sorted(data)
                p95_idx  = min(int(math.ceil(0.95 * n)) - 1, n - 1)
                p95      = sorted_d[p95_idx]
                result[f] = dict(mean=mean, max=mx, min=mn, p95=p95, n=n)
        return result


# ─── 工具函数 ───────────────────────────────────────────────────────────────
def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return DIM + '   --  ' + RESET
    color = GREEN if v < 0.05 else (YELLOW if v < 0.15 else RED)
    return f"{color}{v*1000:7.1f}{RESET}"


def _stat_row(label: str, d: Optional[dict], unit_ms: bool = True) -> str:
    if d is None:
        return f"  {label:<30s}  {'N/A':>8}"
    conv = 1000.0 if unit_ms else 1.0
    unit = 'ms' if unit_ms else ''
    mean_s = _fmt_ms(d['mean']) if unit_ms else f"{d['mean']:7.1f}"
    p95_s  = _fmt_ms(d['p95'])  if unit_ms else f"{d['p95']:7.1f}"
    max_s  = _fmt_ms(d['max'])  if unit_ms else f"{d['max']:7.1f}"
    return (f"  {label:<30s}  mean={mean_s}{unit}  "
            f"p95={p95_s}{unit}  max={max_s}{unit}  (n={d['n']})")


def _section(title: str, node: str, snap: dict):
    lines = []
    lines.append(f"\n{BOLD}{CYAN}── {title} [{node}] ──{RESET}  "
                 f"(received {snap['recv_count']} msgs, window={WINDOW_SIZE})")
    lines.append(_stat_row('total_time',                snap['total_time']))
    lines.append(_stat_row('e2e_latency',              snap['e2e_latency']))
    lines.append(_stat_row('  ├ model_time',            snap['model_time']))
    lines.append(_stat_row('  ├ preprocess_time',       snap['preprocess_time']))
    lines.append(_stat_row('  └ postprocess_time',      snap['postprocess_time']))
    lines.append(_stat_row('avg_model/obstacle',        snap['avg_model_time_per_obstacle']))
    if node == 'aggregator':
        lines.append(_stat_row('sync_wait_time',        snap['sync_wait_time']))
    lines.append(_stat_row('obstacle_count (per frame)',snap['obstacle_count'], unit_ms=False))
    return '\n'.join(lines)


# ─── 主节点 ─────────────────────────────────────────────────────────────────
class RematchProfiler:
    def __init__(self):
        rospy.init_node('rematch_profiler', anonymous=True)
        self.stats: dict[str, NodeStats] = {n: NodeStats() for n in NODES}
        self._shutdown = False

        for node, topic in TOPICS.items():
            rospy.Subscriber(topic, RematchStatistics,
                             lambda msg, n=node: self.stats[n].update(msg),
                             queue_size=20)

        rospy.loginfo(f"[Profiler] Subscribed to: {list(TOPICS.values())}")

        # 定时刷新线程
        self._timer = threading.Thread(target=self._refresh_loop, daemon=True)
        self._timer.start()

        # Ctrl-C 钩子
        signal.signal(signal.SIGINT, self._on_shutdown)

    def _refresh_loop(self):
        rate = rospy.Rate(REFRESH_HZ)
        while not self._shutdown and not rospy.is_shutdown():
            self._print_dashboard()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    def _print_dashboard(self):
        # 清屏
        sys.stdout.write('\033[2J\033[H')

        lines = []
        lines.append(f"{BOLD}{'='*65}{RESET}")
        lines.append(f"{BOLD}  rematch_profiler_live  —  sliding window = {WINDOW_SIZE} frames{RESET}")
        lines.append(f"{BOLD}{'='*65}{RESET}")
        lines.append(f"  {DIM}All times in milliseconds (ms){RESET}")

        for node in NODES:
            snap = self.stats[node].snapshot()
            lines.append(_section(node.upper(), node, snap))

        # 简易瓶颈提示
        lines.append(f"\n{BOLD}── Bottleneck Hint ──{RESET}")
        bottleneck = self._bottleneck_hint()
        lines.append(f"  {bottleneck}")

        lines.append(f"\n{DIM}  Press Ctrl-C to stop and print summary report.{RESET}")
        print('\n'.join(lines))
        sys.stdout.flush()

    def _bottleneck_hint(self) -> str:
        candidates = {}
        for node in ['sam', 'clip']:
            snap = self.stats[node].snapshot()
            if snap['model_time'] is not None:
                candidates[node] = snap['model_time']['mean']
        if not candidates:
            return DIM + "waiting for data..." + RESET
        worst = max(candidates, key=candidates.get)
        val   = candidates[worst]
        color = RED if val > 0.2 else (YELLOW if val > 0.08 else GREEN)
        other = [n for n in candidates if n != worst]
        hint  = (f"{color}{worst.upper()} model_time mean = {val*1000:.1f} ms{RESET}"
                 " is the main bottleneck.")
        if other:
            o_val = candidates[other[0]]
            ratio = val / o_val if o_val > 1e-6 else float('inf')
            hint += f"  ({ratio:.1f}× slower than {other[0].upper()})"
        return hint

    def _on_shutdown(self, *_):
        self._shutdown = True
        self._print_summary()
        rospy.signal_shutdown("user interrupt")

    def _print_summary(self):
        print(f"\n\n{BOLD}{'='*65}{RESET}")
        print(f"{BOLD}  FULL SESSION SUMMARY{RESET}")
        print(f"{BOLD}{'='*65}{RESET}")
        for node in NODES:
            summ = self.stats[node].summary()
            print(_section(f"{node.upper()} (all frames)", node, summ))

        # 输出 CSV 供后续分析
        csv_path = '/tmp/rematch_profiler_summary.csv'
        try:
            with open(csv_path, 'w') as f:
                fields = NodeStats.FIELDS
                f.write('node,' + ','.join(
                    [f'{field}_{stat}' for field in fields
                     for stat in ['mean', 'p95', 'max', 'min']]) + '\n')
                for node in NODES:
                    summ = self.stats[node].summary()
                    row = [node]
                    for field in fields:
                        d = summ.get(field)
                        if d is None:
                            row += ['', '', '', '']
                        else:
                            row += [f"{d['mean']:.6f}", f"{d['p95']:.6f}",
                                    f"{d['max']:.6f}", f"{d['min']:.6f}"]
                    f.write(','.join(row) + '\n')
            print(f"\n{GREEN}  Summary CSV saved to: {csv_path}{RESET}")
        except Exception as e:
            print(f"{RED}  Failed to save CSV: {e}{RESET}")

    def spin(self):
        rospy.spin()


def main():
    profiler = RematchProfiler()
    profiler.spin()


if __name__ == '__main__':
    main()
