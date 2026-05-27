"""Schedule table QA — time overlap, resource allocation, duration."""
import random
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context
from ._mcq_letter_lib import maybe_to_mcq_letter

_Q_LONGEST = [
    "Which event has the longest duration?",
    "In the schedule, which event runs for the most hours?",
    "Report the event that lasts the longest.",
    "Which event takes the greatest duration in this schedule?",
    "Based on the table, which event has the maximum duration?",
    "Identify the single event with the longest duration.",
    "Looking at the table, which event is scheduled for the most hours?",
    "Name the event with the longest scheduled duration.",
    "Which row in the schedule corresponds to the event with the greatest duration?",
    "Find the event whose duration exceeds all others.",
    "Which event has the highest Duration value in the schedule?",
    "Pick out the event with the longest time span from the schedule.",
    "In the table, which event uses the most hours?",
    "Which event has the longest-running scheduled block?",
    "State the event with the longest duration in the schedule table.",
    "Which entry in the schedule has the largest Duration value?",
]

_Q_OVERLAP = [
    "How many pairs of events overlap in time?",
    "Count the number of event pairs with overlapping time ranges.",
    "How many pairs of events have time intervals that intersect?",
    "Report the count of pairs (i, j) whose schedules overlap.",
    "How many scheduled events share time with another event? Give the pair count.",
    "Count overlapping event pairs in the schedule.",
    "In the schedule, how many pairs of events run simultaneously at some point?",
    "Report the number of event pairs that conflict in time.",
    "How many pairs of events have overlapping start/end windows?",
    "Give the total count of time-overlapping event pairs.",
    "How many event-pairs share overlapping time slots in the schedule?",
    "Count the pairs of events with time overlap in the table.",
    "From the schedule, how many pairs of events run at the same time?",
    "How many pairs (i, j) of events in the schedule overlap?",
    "Tally the number of scheduling conflicts (overlapping pairs) in the table.",
    "How many pairs of events have intersecting time intervals?",
]

_Q_TOTAL_HRS = [
    "What is the total number of hours across all events?",
    "Sum the durations of all events. What is the total in hours?",
    "Give the combined total duration (hours) of the events.",
    "How many total hours are scheduled across all events?",
    "Sum up the Duration column. What total do you get?",
    "What is the aggregate duration (in hours) of the schedule?",
    "Compute the sum of all event durations in the schedule.",
    "What is the total scheduled time across all events (in hours)?",
    "Add the hours of every event. What is the sum?",
    "Report the grand total of the Duration column.",
    "Across all events in the table, what is the total hours?",
    "What is the sum of all durations in the schedule?",
    "Give the total scheduled hours across the schedule.",
    "Sum the Duration values to find the total hours scheduled.",
    "What is the combined hour-count of all events in the schedule?",
    "Total hours scheduled = ? Add every event's duration.",
]

_Q_FREE_GAP = [
    "What is the longest gap (in hours) between consecutive events?",
    "Report the largest gap (hours) between adjacent scheduled events.",
    "Between consecutive events, what is the maximum idle gap in hours?",
    "Find the longest free interval between sequential events (in hours).",
    "What is the maximum hour-gap between two consecutive events?",
    "Across consecutive-event pairs, which gap is longest? Give hours.",
    "Report the longest quiet gap (in hours) between adjacent events.",
    "What is the longest idle-time gap between consecutive events in hours?",
    "How long is the biggest gap (hours) between the end of one event and the start of the next?",
    "Give the maximum gap in hours between consecutive events.",
    "What's the largest quiet stretch (hours) between adjacent events?",
    "Between sequential events in the schedule, what is the longest gap (hours)?",
    "Find the longest down-time (in hours) between consecutive scheduled events.",
    "Report the size of the biggest empty gap (hours) in the schedule.",
    "What is the longest interval (in hours) where no event is scheduled between two adjacent events?",
    "Between two consecutive events, what's the maximum free stretch (hours)?",
]

_Q_MAX_CONC = [
    "What is the maximum number of events occurring at the same time?",
    "At the busiest moment, how many events run simultaneously?",
    "What's the peak concurrency of events in the schedule?",
    "Report the maximum count of simultaneous events at any single time.",
    "How many events overlap at the peak moment in the schedule?",
    "At the busiest hour, how many events are active?",
    "Give the maximum number of events active at the same time.",
    "What is the maximum concurrency (simultaneous events) in this schedule?",
    "At peak time, how many events run at once?",
    "Count the largest number of events overlapping at any single time.",
    "Report the highest concurrency reached in the schedule.",
    "What's the maximum number of overlapping events in the schedule?",
    "At the busiest point, how many events run concurrently?",
    "How many events coincide at the schedule's peak hour?",
    "What is the peak number of concurrently-running events?",
    "Give the max number of events running simultaneously in the schedule.",
]

_Q_EARLIEST_FREE = [
    "Find the earliest 1-hour slot (starting from {start}:00) where no events are scheduled. Give the start time as an integer (e.g. 10 for 10:00).",
    "Starting from {start}:00, what is the earliest integer hour with no scheduled event? Integer only.",
    "At what earliest integer hour (starting from {start}:00) is there a free 1-hour slot? Report the integer.",
    "From {start}:00 onward, give the earliest integer hour with a 1-hour vacancy. Integer answer.",
    "Report the earliest 1-hour free slot starting time (integer hour, from {start}:00).",
    "What is the first integer hour from {start}:00 where no event overlaps? Integer answer.",
    "Starting at {start}:00, find the earliest free 1-hour window start-time (integer).",
    "Locate the earliest integer hour (from {start}:00) with no event scheduled. Integer answer.",
    "From hour {start}:00, what's the first unscheduled integer hour? Integer only.",
    "Report the earliest integer hour on or after {start}:00 with a 1-hour free slot.",
    "From {start}:00, find the earliest integer hour that has no scheduled event.",
    "What's the earliest integer hour (starting at {start}:00) where no events overlap? Integer.",
    "Starting from {start}:00, give the earliest integer hour with a 1-hour gap. Integer.",
    "Find the first integer hour on or after {start}:00 with no event scheduled. Integer only.",
    "Starting at {start}:00, what integer hour is the earliest 1-hour free window? Integer answer.",
    "From the schedule, what's the earliest integer hour (from {start}:00) with a free 1-hour slot?",
]

class ScheduleTableQA(StandaloneVisualEnv):
    ENV_NAME = "schedule_table"

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"qtypes": ["longest", "total_hours"],
                    "qweights": [5, 5], "n_events": (3, 4), "dur_range": (1, 2)}
        if level <= 2:
            return {"qtypes": ["longest", "total_hours", "free_gap"],
                    "qweights": [3, 4, 3], "n_events": (4, 5), "dur_range": (1, 3)}
        if level <= 4:
            return {"qtypes": ["overlap_count", "free_gap", "total_hours"],
                    "qweights": [4, 3, 3], "n_events": (4, 6), "dur_range": (1, 3)}
        if level <= 6:
            return {"qtypes": ["overlap_count", "free_gap", "max_concurrent_events"],
                    "qweights": [3, 3, 4], "n_events": (5, 7), "dur_range": (1, 3)}
        if level <= 8:
            return {"qtypes": ["max_concurrent_events", "earliest_free_slot", "overlap_count"],
                    "qweights": [3, 4, 3], "n_events": (5, 7), "dur_range": (1, 4)}
        return {"qtypes": ["earliest_free_slot", "max_concurrent_events"],
                "qweights": [5, 5], "n_events": (6, 8), "dur_range": (1, 4)}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        # Retry with slightly-varied sub_seed to escape "longest" ties.
        for attempt in range(8):
            r = self._try_generate(seed, parameter, attempt)
            if r is not None:
                return r
        return None

    def _try_generate(self, seed: int, parameter: Dict, attempt: int) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 6601 + attempt * 131)

        n_events = sub_rng.randint(*cfg["n_events"])
        events = [f"Event {chr(65+i)}" for i in range(n_events)]
        starts = sorted([sub_rng.randint(8, 16) for _ in range(n_events)])
        durations = [sub_rng.randint(*cfg["dur_range"]) for _ in range(n_events)]
        ends = [s + d for s, d in zip(starts, durations)]

        question_type = parameter.get("question_type")
        if question_type not in ["longest", "overlap_count", "total_hours", "free_gap",
                                   "max_concurrent_events", "earliest_free_slot"]:
            question_type = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        sidx = (self.seed or 0) % 16
        if question_type == "longest":
            max_d = max(durations)
            top_idx = [i for i, d in enumerate(durations) if d == max_d]
            if len(top_idx) != 1:
                return None
            idx = top_idx[0]
            question = _Q_LONGEST[sidx]
            answer = events[idx]
        elif question_type == "overlap_count":
            overlaps = 0
            for i in range(n_events):
                for j in range(i+1, n_events):
                    if starts[i] < ends[j] and starts[j] < ends[i]:
                        overlaps += 1
            question = _Q_OVERLAP[sidx]
            answer = overlaps
        elif question_type == "total_hours":
            total = sum(durations)
            question = _Q_TOTAL_HRS[sidx]
            answer = total
        elif question_type == "free_gap":
            gaps = []
            sorted_events = sorted(zip(starts, ends, events))
            for i in range(len(sorted_events)-1):
                gap = sorted_events[i+1][0] - sorted_events[i][1]
                if gap > 0:
                    gaps.append(gap)
            answer = max(gaps) if gaps else 0
            question = _Q_FREE_GAP[sidx]
        elif question_type == "max_concurrent_events":
            max_conc = 0
            for t in range(min(starts), max(ends) + 1):
                count = sum(1 for i in range(n_events) if starts[i] <= t < ends[i])
                max_conc = max(max_conc, count)
            question = _Q_MAX_CONC[sidx]
            answer = max_conc
        elif question_type == "earliest_free_slot":
            earliest_free = None
            for t in range(min(starts), max(ends) + 2):
                busy = any(starts[i] <= t < ends[i] for i in range(n_events))
                if not busy:
                    earliest_free = t
                    break
            if earliest_free is None:
                earliest_free = max(ends)
            question = _Q_EARLIEST_FREE[sidx].format(start=min(starts))
            answer = earliest_free
        else:
            return None

        # Render as table with mode triad + color/font variation
        style = self._random_style()
        mode = pick_render_mode(sub_rng)
        if mode == "textbook":
            tbp = textbook_params(sub_rng)
            bg = tbp["bg"]
            header_bg = tbp["line_color"]
            body_bg = tbp["fill_color"]
            edge = tbp["aux_color"]
            dpi = tbp["dpi"]
            font_kw = {"fontfamily": tbp["font_family"]}
        elif mode == "sketch":
            bg = sub_rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            header_bg = style["palette"][0]
            body_bg = "#f8f9fa"
            edge = "#1a1a1a"
            dpi = style["dpi"]
            font_kw = {}
        else:
            bg = style["bg_color"]
            header_bg = style["palette"][0]
            body_bg = "#f8f9fa"
            edge = "#7f8c8d"
            dpi = style["dpi"]
            font_kw = {}

        def _draw():
            fig, ax = plt.subplots(figsize=(7, max(3, n_events * 0.6 + 1.5)))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            ax.axis("off")

            cell_text = [[events[i], f"{starts[i]}:00", f"{ends[i]}:00", f"{durations[i]}h"]
                         for i in range(n_events)]
            table = ax.table(cellText=cell_text,
                             colLabels=["Event", "Start", "End", "Duration"],
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(style["font_size_base"])
            table.scale(1, 1.5)

            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_facecolor(header_bg)
                    cell.set_text_props(color='white', fontweight='bold', **font_kw)
                else:
                    cell.set_facecolor(body_bg)
                    cell.set_text_props(**font_kw)
                cell.set_edgecolor(edge)

            ax.set_title("Schedule", fontsize=style["font_size_base"] + 2,
                         fontweight="bold", pad=20, **font_kw)
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=70, randomness=1.3):
                fig = _draw()
        else:
            fig = _draw()

        # MCQ-letter style MCQ-letter mode: with prob 0.5, convert to MCQ.
        # For "longest" mode the answer is an event label — use other event
        # labels as candidate pool. Otherwise it's numeric.
        ans_str = str(answer)
        n_opts = sub_rng.choice([4, 5])
        if question_type == "longest":
            cand_pool = list(events)
        else:
            cand_pool = None
        question, ans_str = maybe_to_mcq_letter(
            question, ans_str, sub_rng, prob=1.0, n_options=n_opts,
            candidate_pool=cand_pool)
        return question, ans_str, self.fig_to_pil(fig, dpi=dpi)
