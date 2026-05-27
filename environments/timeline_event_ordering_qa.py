"""
Timeline Event Ordering QA environment.

Capabilities: D1 (value extraction) + L3 (temporal ordering)
Target regression: dynamic-math statistics, spatial-reasoning Motion-Cam.

A horizontal timeline with 4-8 labeled events (dots with text labels).
Each event sits at a specific position corresponding to a date / year /
tick value. Timeline spacing can be uniform, log-scaled, or arbitrary
irregular. Labels may be slightly crowded.

4-option MCQ asking about ordering, closest pair, or gaps between events.

Difficulty schedule (0..9):
  Axis 1: n_events = 4 + level // 2      -> 4..8
  Axis 2: timeline_spacing L<=3: uniform
          L4..L6: log-scale
          L>=7: arbitrary irregular
  Axis 3: label_crowding = level >= 4

Output: (question_str, answer_letter, PIL_Image)
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_EVENT_NAME_POOLS = [
    ["Launch", "Beta", "v1.0", "v2.0", "Patch", "Update", "Final", "Release"],
    ["Kickoff", "Planning", "Design", "Build", "QA", "Pilot",
     "Rollout", "Close"],
    ["Event A", "Event B", "Event C", "Event D", "Event E",
     "Event F", "Event G", "Event H"],
    ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5",
     "Phase 6", "Phase 7", "Phase 8"],
    ["Discovery", "Hypothesis", "Experiment", "Analysis", "Publication",
     "Review", "Citation", "Follow-up"],
]

class TimelineEventOrderingQA(StandaloneVisualEnv):
    ENV_NAME = "timeline_event_ordering"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # L9 hardening (2026-04-17): larger event set and new harder qtypes
        # (events_between_range, event_after_gap). Events grow 4..10.
        n_events = min(10, 4 + level // 2 + (2 if level >= 8 else 0))
        if level <= 3:
            spacing = "uniform"
        elif level <= 6:
            spacing = "log"
        else:
            spacing = "irregular"
        label_crowding = level >= 4
        if level <= 2:
            qtype_pool = ["nth_event"]
        elif level <= 6:
            qtype_pool = ["gap_between"]
        elif level <= 7:
            qtype_pool = ["closest_pair"]
        else:
            # L8-L9: add harder qtypes — count events in a date window, and
            # find the event that occurred a given number of years after
            # another event. Mixed at L8-L9.
            qtype_pool = ["events_in_window", "event_after_gap",
                           "closest_pair"]
        return {
            "n_events": n_events,
            "spacing": spacing,
            "label_crowding": label_crowding,
            "qtype_pool": qtype_pool,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2273)

        for _ in range(10):
            try:
                result = self._try_generate(sub_rng, cfg, level)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_events"]
        pool = list(rng.choice(_EVENT_NAME_POOLS))
        rng.shuffle(pool)
        events = pool[:n]

        # Generate positions (strictly increasing)
        t_start = rng.randint(2000, 2015)
        if cfg["spacing"] == "uniform":
            step = rng.randint(1, 3)
            positions = [t_start + i * step for i in range(n)]
        elif cfg["spacing"] == "log":
            # Early events cluster tightly; later events spread
            base_gaps = [round(1.5 ** i, 1) for i in range(n - 1)]
            positions = [float(t_start)]
            for g in base_gaps:
                positions.append(round(positions[-1] + g, 1))
        else:  # irregular
            positions = [float(t_start)]
            for _ in range(n - 1):
                gap = round(rng.uniform(0.3, 5.5), 1)
                positions.append(round(positions[-1] + gap, 1))

        qtype = rng.choice(cfg["qtype_pool"])

        if qtype == "nth_event":
            # Pick a random ordinal: "Which event occurred 3rd?"
            k = rng.randint(1, n)
            correct_name = events[k - 1]
            ordinal = self._ordinal(k)
            q_stem = f"Which event occurred {ordinal} on the timeline?"
            options = list(events)
            rng.shuffle(options)
            correct_idx = options.index(correct_name)
            letter = chr(ord("A") + correct_idx)
            opts_str = " ".join(
                f"({chr(ord('A')+i)}) {v}" for i, v in enumerate(options)
            )
            question = f"{q_stem} Options: {opts_str}. Answer with a single letter."

        elif qtype == "gap_between":
            # Ask the time gap between two chosen events
            i, j = sorted(rng.sample(range(n), 2))
            gap = round(positions[j] - positions[i], 1)
            # Require nontrivial gap
            if gap <= 0:
                return None
            q_stem = (f"How much time elapsed between '{events[i]}' and "
                      f"'{events[j]}' on the timeline?")
            correct_disp = gap
            # Distractors: nearby gaps
            base = max(0.5, gap * 0.3)
            offsets = [-base, base, -2 * base, 2 * base, -0.5 * base,
                       0.5 * base]
            rng.shuffle(offsets)
            distractors = []
            for off in offsets:
                cand = round(correct_disp + off, 1)
                if cand <= 0 or abs(cand - correct_disp) < 0.2:
                    continue
                if any(abs(cand - d) < 0.1 for d in distractors):
                    continue
                distractors.append(cand)
                if len(distractors) >= 3:
                    break
            if len(distractors) < 3:
                return None
            options = [correct_disp] + distractors[:3]
            rng.shuffle(options)
            correct_idx = options.index(correct_disp)
            letter = chr(ord("A") + correct_idx)
            opts_str = " ".join(
                f"({chr(ord('A')+i)}) {v}" for i, v in enumerate(options)
            )
            question = f"{q_stem} Options: {opts_str}. Answer with a single letter."

        elif qtype == "closest_pair":
            # Find the pair of events closest in time
            best = None
            best_gap = float("inf")
            for i in range(n - 1):
                gap = positions[i + 1] - positions[i]
                if gap < best_gap:
                    best_gap = gap
                    best = i
            # Require all pair gaps distinct enough
            all_gaps = [positions[i + 1] - positions[i] for i in range(n - 1)]
            sorted_gaps = sorted(all_gaps)
            if len(sorted_gaps) >= 2 and sorted_gaps[0] * 1.15 > sorted_gaps[1]:
                return None
            correct_pair = f"{events[best]} and {events[best + 1]}"
            # Build 3 other adjacent pairs as distractors
            other_pairs = [(i, f"{events[i]} and {events[i+1]}")
                           for i in range(n - 1) if i != best]
            rng.shuffle(other_pairs)
            distractors = [p for _, p in other_pairs[:3]]
            if len(distractors) < 3:
                return None
            options = [correct_pair] + distractors
            rng.shuffle(options)
            correct_idx = options.index(correct_pair)
            letter = chr(ord("A") + correct_idx)
            opts_str = " ".join(
                f"({chr(ord('A')+i)}) {v}" for i, v in enumerate(options)
            )
            q_stem = ("Looking at the timeline, which pair of consecutive "
                       "events are closest to each other in time?")
            question = f"{q_stem} Options: {opts_str}. Answer with a single letter."

        elif qtype == "events_in_window":
            # Pick a window [t_lo, t_hi] that includes some events.
            # Ask how many events are strictly inside the window.
            if n < 5:
                return None
            # Choose a window covering 2..(n-2) events
            target_count = rng.randint(2, max(2, n - 2))
            # Pick a start index so positions[start : start+target_count] lie inside
            start = rng.randint(0, n - target_count)
            end = start + target_count - 1
            # Window lower is just above positions[start-1] (or positions[start] - margin)
            # Window upper is just below positions[end+1] (or positions[end] + margin)
            if start == 0:
                lo_win = round(positions[0] - 0.3, 1)
            else:
                lo_win = round((positions[start - 1] + positions[start]) / 2.0, 1)
            if end == n - 1:
                hi_win = round(positions[n - 1] + 0.3, 1)
            else:
                hi_win = round((positions[end] + positions[end + 1]) / 2.0, 1)
            # Recount strictly inside
            cnt = sum(1 for p in positions if lo_win <= p <= hi_win)
            if cnt != target_count:
                return None
            q_stem = (f"How many events on the timeline fall within the window "
                      f"[{lo_win}, {hi_win}] (inclusive)?")
            correct_disp = cnt
            # Distractors: +-1, +-2 (must be positive, within n, distinct)
            distractors = []
            for off in [1, -1, 2, -2, 3]:
                cand = correct_disp + off
                if cand < 0 or cand > n:
                    continue
                if cand == correct_disp:
                    continue
                if cand in distractors:
                    continue
                distractors.append(cand)
                if len(distractors) >= 3:
                    break
            if len(distractors) < 3:
                return None
            options = [correct_disp] + distractors[:3]
            rng.shuffle(options)
            correct_idx = options.index(correct_disp)
            letter = chr(ord("A") + correct_idx)
            opts_str = " ".join(
                f"({chr(ord('A')+i)}) {v}" for i, v in enumerate(options)
            )
            question = f"{q_stem} Options: {opts_str}. Answer with a single letter."

        elif qtype == "event_after_gap":
            # "Which event occurred approximately G years/units after 'E'?"
            # where G ≈ positions[k] - positions[i] for some i,k pair.
            if n < 5:
                return None
            i = rng.randint(0, n - 3)
            k = rng.randint(i + 2, n - 1)
            gap = round(positions[k] - positions[i], 1)
            if gap < 1.5:
                return None
            target_name = events[k]
            src_name = events[i]
            q_stem = (f"Which event occurred approximately {gap} units after "
                      f"'{src_name}' on the timeline?")
            # Options: target + 3 other non-src, non-target events
            other_events = [events[j] for j in range(n)
                            if events[j] != target_name and events[j] != src_name]
            rng.shuffle(other_events)
            distractors = other_events[:3]
            if len(distractors) < 3:
                return None
            options = [target_name] + distractors
            rng.shuffle(options)
            correct_idx = options.index(target_name)
            letter = chr(ord("A") + correct_idx)
            opts_str = " ".join(
                f"({chr(ord('A')+i)}) {v}" for i, v in enumerate(options)
            )
            question = f"{q_stem} Options: {opts_str}. Answer with a single letter."

        else:
            return None

        image = self._render(rng, events, positions, cfg)
        return question, letter, image

    @staticmethod
    def _ordinal(k):
        if 10 <= k % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
        return f"{k}{suffix}"

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, rng, events, positions, cfg):
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)

        # Compute figure size based on range
        n = len(events)
        fig_w = max(8.0, 1.0 + n * 1.2) * style["figsize_scale"]
        fig_h = 3.5 * style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        x_min = min(positions)
        x_max = max(positions)
        x_pad = max(0.5, (x_max - x_min) * 0.08)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(-1.2, 1.6)
        ax.axis("off")

        # Draw main timeline
        line_color = style["geo_line_color"]
        ax.plot([x_min - x_pad * 0.5, x_max + x_pad * 0.5],
                [0, 0], color=line_color,
                linewidth=max(2.0, style["line_width"]))
        # Tick marks at min/max
        ax.plot([x_min, x_max], [0, 0], "o", color=line_color, markersize=1)

        # Draw events as dots with labels. Alternate above/below if crowding.
        fontsize = style["font_size_base"]
        for i, (name, pos) in enumerate(zip(events, positions)):
            color = palette[i % len(palette)]
            ax.plot(pos, 0, "o", color=color, markersize=11,
                    markeredgecolor="black", zorder=4)
            # Alternating label height helps but may still crowd at L>=4
            if cfg["label_crowding"]:
                height = 0.55 if i % 2 == 0 else -0.55
                tick_y = 0.25 if i % 2 == 0 else -0.25
            else:
                height = 0.55
                tick_y = 0.25
            ax.plot([pos, pos], [0, tick_y], color=color, linewidth=1.2)
            ax.text(pos, height, name, ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=fontsize, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2",
                               fc=style["bg_color"], ec=color, alpha=0.85))
            # Position tick label under line
            ax.text(pos, -0.95 if not cfg["label_crowding"] or i % 2 == 0 else 0.95,
                    f"{pos:g}", ha="center",
                    va="top" if not cfg["label_crowding"] or i % 2 == 0 else "bottom",
                    fontsize=fontsize - 2, color="#555555")

        ax.set_title("Timeline of Events",
                     fontsize=style["font_size_base"] + 2, pad=6)
        fig.patch.set_facecolor(style["bg_color"])
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
