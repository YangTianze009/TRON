"""
Calendar Date QA (D35, P1).

Reference an external reference — calendar date / weekday lookup.

Renders a monthly calendar (with month name and year) where the dates are
laid out in standard 7-column grid (Sunday or Monday first). One date is
highlighted; the model is asked for its weekday name (e.g., "Wednesday"),
or the date a given number of days later/earlier.

Verifier: weekday string or date string. Verifier already handles substring
match for word answers via base class.
"""
import math
import random
import calendar
from datetime import date, timedelta
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]


class CalendarDateQA(StandaloneVisualEnv):
    ENV_NAME = "calendar_date"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"modes": ["weekday_of_date"], "year_range": (2020, 2026)}
        if level <= 5:
            return {"modes": ["weekday_of_date", "date_offset"],
                    "year_range": (2018, 2028)}
        return {"modes": ["weekday_of_date", "date_offset"],
                "year_range": (2010, 2030)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 6203 + level * 89 + 23)

        mode = rng.choice(cfg["modes"])
        year = rng.randint(*cfg["year_range"])
        month = rng.randint(1, 12)
        days_in_month = calendar.monthrange(year, month)[1]
        target_day = rng.randint(1, days_in_month)
        target_date = date(year, month, target_day)

        if mode == "weekday_of_date":
            weekday_idx = target_date.weekday()
            answer = _WEEKDAYS[weekday_idx]
            question = (
                f"The figure shows the calendar of {_MONTH_NAMES[month - 1]} "
                f"{year}. The highlighted date is {target_day}. What day of "
                f"the week is {_MONTH_NAMES[month - 1]} {target_day}, {year}? "
                f"(Answer with the weekday name, e.g., Monday.)"
            )
        else:  # date_offset
            offset = rng.choice([-30, -14, -7, 7, 14, 30])
            new_date = target_date + timedelta(days=offset)
            answer = f"{_MONTH_NAMES[new_date.month - 1]} {new_date.day}, {new_date.year}"
            sign_word = "after" if offset > 0 else "before"
            question = (
                f"The figure shows the calendar of {_MONTH_NAMES[month - 1]} "
                f"{year}. The highlighted date is {target_day}. What date is "
                f"{abs(offset)} days {sign_word} {_MONTH_NAMES[month - 1]} "
                f"{target_day}, {year}? Answer in the format `Month Day, Year` "
                f"(e.g., March 15, 2025)."
            )

        img = self._render(year, month, target_day)
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _render(self, year, month, target_day) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        ax.text(3.5, 6.5, f"{_MONTH_NAMES[month - 1]} {year}",
                ha="center", fontsize=18, fontweight="bold", color="#2c3e50")

        # Header
        # We use Monday-first convention to match `calendar.monthrange`
        days_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, d in enumerate(days_short):
            ax.text(i + 0.5, 5.5, d, ha="center", fontsize=11,
                    fontweight="bold", color="#7f8c8d")

        # Generate month grid
        cal = calendar.monthcalendar(year, month)  # weeks list, Monday-first
        for week_idx, week in enumerate(cal):
            for day_idx, day in enumerate(week):
                if day == 0:
                    continue
                x = day_idx + 0.5
                y = 4.5 - week_idx
                if day == target_day:
                    ax.add_patch(patches.Circle((x, y), 0.35,
                                                 color="#fce8e6",
                                                 ec="#d62728", linewidth=2,
                                                 zorder=2))
                    ax.text(x, y, str(day), ha="center", va="center",
                            fontsize=12, fontweight="bold", color="#d62728")
                else:
                    ax.text(x, y, str(day), ha="center", va="center",
                            fontsize=11, color="#2c3e50")

        # Grid lines
        for i in range(8):
            ax.plot([i, i], [-1, 6], color="#bdc3c7", linewidth=0.5)
        for j in range(-1, 6):
            ax.plot([0, 7], [j, j], color="#bdc3c7", linewidth=0.5)

        ax.set_xlim(-0.2, 7.2)
        ax.set_ylim(-1, 7)
        ax.axis("off")
        ax.set_aspect("auto")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = CalendarDateQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   positive={v_pos['accuracy']} negative={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
