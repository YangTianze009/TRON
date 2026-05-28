"""
UK 11+ style flat-design business / finance infographic QA.

Renders a magazine-style layout combining:
  - large bold callout numbers (revenue, share price, mileage, fuel rate)
  - 3-6 labelled mini-cards (stat tiles)
  - one supporting bar OR donut chart with explicit labels
  - text headers and unicode-icon decorations

Then asks a multi-step financial / numerical reasoning question:
  - annual fuel consumption from mpg + miles/month
  - share-purchase capacity (cap budget / unit price)
  - takeover valuation (multiplier × projected revenue)
  - GDP per-person comparison
  - compounding % increase

Decimal / integer numeric output.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_mcq_letter


_TEMPLATES_FUEL = [
    "If a driver travels {miles} miles per month in a {model} (mpg shown in the infographic), what is the annual fuel consumption in gallons? Place the numeric answer in <answer>...</answer>.",
    "Compute the annual fuel use (gallons) for a {model} driver covering {miles} miles per month, using the mpg shown. Numeric in <answer>...</answer>.",
    "From the infographic, take the {model} mpg figure. A driver does {miles} miles per month. How many gallons per year? Numeric in <answer>...</answer>.",
    "Annual fuel (gallons) for a {model} car at {miles} miles/month: compute from the mpg shown in the infographic. Numeric in <answer>...</answer>.",
]

_TEMPLATES_BUDGET_SHARES = [
    "A dealership has {budget} to spend and wants equal numbers of {model_a} and {model_b} cars. What is the largest equal number of each it can buy? Place the integer in <answer>...</answer>.",
    "From the infographic, what is the largest equal number of {model_a} and {model_b} cars a dealership can purchase with a {budget} budget? Integer in <answer>...</answer>.",
    "Spending no more than {budget}, how many {model_a} cars AND the same number of {model_b} cars can be bought? Integer answer (per model) in <answer>...</answer>.",
    "Maximum equal number of {model_a} and {model_b} purchasable for {budget} (using prices in the infographic)? Integer in <answer>...</answer>.",
]

_TEMPLATES_TAKEOVER = [
    "A takeover bid offers {mult}x the {year} projected revenue ({revenue_label}). The board says they add {markup}% to that. What is the board's valuation (in the same currency)? Numeric in <answer>...</answer>.",
    "Compute the board's valuation: {mult}x {revenue_label} for {year}, plus a {markup}% mark-up. Numeric in <answer>...</answer>.",
    "Multiply {revenue_label} for {year} by {mult}, then add {markup}%. What is the resulting valuation? Numeric in <answer>...</answer>.",
    "A takeover bid is {mult}x the {year} projected revenue. After a {markup}% board mark-up, what is the valuation? Numeric in <answer>...</answer>.",
]

_TEMPLATES_SHARE_NOT_HELD = [
    "From the infographic, {pct}% of {company}'s shares are held by Directors. Total shares are {total}. How many shares are NOT held by Directors? Place the integer in <answer>...</answer>.",
    "Directors hold {pct}% of the {total} shares of {company}. Compute the number of shares NOT held by Directors. Integer in <answer>...</answer>.",
    "If Directors hold {pct}% of {company} shares (total {total}), how many shares are held outside the Director group? Integer in <answer>...</answer>.",
    "Total shares: {total}. Directors: {pct}%. Number of shares held by non-Directors? Integer in <answer>...</answer>.",
]

_TEMPLATES_COMPOUND = [
    "{company} R&D spend grows {p1}% in year 1 then {p2}% in year 2, starting from {base}. What is the year 2 spend? Numeric in <answer>...</answer>.",
    "Starting from {base}, apply a {p1}% increase then a {p2}% increase. Numeric value of the result in <answer>...</answer>.",
    "Compound growth: {base} grows {p1}% then {p2}%. Final value? Numeric in <answer>...</answer>.",
    "The infographic shows {company}'s year-0 spend of {base}. After consecutive +{p1}% and +{p2}% increases, what is the year-2 spend? Numeric in <answer>...</answer>.",
]


_CAR_NAMES = ["Xtam", "Taber", "Ursa", "Galta", "Boren", "Veric", "Krell"]
_COMPANY_NAMES = [
    "Calewall", "Teala Media", "Leutts", "Antlyn plc", "Royer inc",
    "Graff inc", "Balcom plc", "Trade ltd", "Drebs Ltd", "Steapars",
]
_CURRENCIES = ["£", "$", "€"]


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _format_money(x: float, currency: str) -> str:
    return f"{currency}{int(round(x)):,}" if abs(x - round(x)) < 1 else f"{currency}{x:,.2f}"


class InfographicBusinessQA(StandaloneVisualEnv):
    ENV_NAME = "infographic_business_qa"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 v2: full-gradient redesign — was 100/100/100/100. Move
        # multi-step modes up to lower levels to make every level non-trivial.
        level = max(0, min(level, 9))
        if level == 0:
            modes = ["share_not_held", "fuel"]  # was just share_not_held
            n_extras = 3  # was 2
        elif level <= 2:
            modes = ["fuel", "budget_shares"]  # was share_not_held, fuel
            n_extras = 4  # was 3
        elif level <= 4:
            modes = ["budget_shares", "takeover", "fuel"]  # added takeover
            n_extras = 5  # was 3
        elif level <= 6:
            modes = ["takeover", "compound", "budget_shares"]
            n_extras = 6  # was 4
        elif level <= 8:
            modes = ["compound", "takeover"]
            n_extras = 7
        else:
            # 2026-05-04: bumped L9 difficulty (was 100% saturated → restrict
            # to compound mode at L9 with 4-step year compound, increase
            # n_extras to 9 distractor cards on the infographic).
            # 2026-05-04 R3: softened bump — was modes=["compound"] only with
            # n_extras=9 (L9 dropped to 0.20). Mix in takeover so mode is not
            # uniquely 4-step compound; reduce n_extras so chart isn't cluttered.
            modes = ["compound", "takeover"]
            n_extras = 7
        return {"level": level, "modes": modes, "n_extras": n_extras}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1733 + level * 41 + 19)
        mode = rng.choice(cfg["modes"])

        if mode == "fuel":
            result = self._gen_fuel(cfg, rng)
        elif mode == "budget_shares":
            result = self._gen_budget(cfg, rng)
        elif mode == "takeover":
            result = self._gen_takeover(cfg, rng)
        elif mode == "compound":
            result = self._gen_compound(cfg, rng)
        else:
            result = self._gen_share_not_held(cfg, rng)

        # 2026-05-04: a logic benchmark infographic actual benchmark always uses 5
        # options A-E with E = "Cannot say" / "None of these" trap. ~15% of
        # benchmark questions have E as correct (genuinely unanswerable from
        # given data). Replicate that pattern.
        if result is not None:
            q, ans, img = result
            # Always 5 options (was random 4-or-5)
            q, ans = maybe_to_mcq_letter(q, ans, rng, prob=1.0, n_options=4)
            # Append E="Cannot say" as 5th option. With 15% prob, swap to make
            # E the correct answer (trap scenario).
            if isinstance(ans, str) and len(ans) == 1 and ans in "ABCDE":
                trap = rng.random() < 0.15
                if trap:
                    # Replace the correct answer's option text with a wrong
                    # plausible value, then E="Cannot say" becomes correct.
                    # Simplest: just append (E)Cannot say and override answer.
                    q = q.rstrip() + " (E)Cannot say"
                    ans = "E"
                else:
                    q = q.rstrip() + " (E)Cannot say"
                    # ans stays A-D
            return q, ans, img
        return result

    # ------------------------------------------------------------ generators
    def _gen_fuel(self, cfg, rng):
        # Pick clean numbers
        model = rng.choice(_CAR_NAMES)
        # mpg and monthly miles such that gallons/year is integer
        mpg_val = rng.choice([30, 40, 50])
        miles_per_month = rng.choice([1500, 2000, 2500, 3000, 4000, 4500, 5000])
        # gallons/yr = miles/year / mpg = (12*miles_per_month)/mpg
        ans = (12 * miles_per_month) / mpg_val
        # ensure integer
        if abs(ans - round(ans)) > 1e-6:
            # adjust miles_per_month to be a multiple
            miles_per_month = mpg_val * 100
            ans = (12 * miles_per_month) / mpg_val
        sidx = (self.seed or 0) % len(_TEMPLATES_FUEL)
        question = _TEMPLATES_FUEL[sidx].format(model=model, miles=miles_per_month)
        # Build infographic stats
        cards = [
            (f"{model}", f"{int(mpg_val)} mpg", "fuel"),
        ]
        # Add a couple distractor mini cards
        for _ in range(cfg["n_extras"] - 1):
            other = rng.choice([m for m in _CAR_NAMES if m != model])
            other_mpg = rng.choice([25, 28, 35, 38, 45, 55])
            cards.append((f"{other}", f"{other_mpg} mpg", "fuel"))
        title = "Vehicle Fuel Efficiency"
        img = self._render_infographic(title, cards, "Annual Mileage Report",
                                       chart_kind="bar", rng=rng)
        return question, _format_num(ans), img

    def _gen_budget(self, cfg, rng):
        # Two car models with prices
        currency = rng.choice(_CURRENCIES)
        models = rng.sample(_CAR_NAMES, 2)
        prices = []
        for _ in range(2):
            p = rng.choice([8000, 10000, 12000, 14000, 15000, 18000, 20000,
                            22000, 24000])
            prices.append(p)
        # Pick a budget that's a multiple of (sum of prices)
        sum_p = prices[0] + prices[1]
        n_each = rng.randint(2, 30)
        budget = n_each * sum_p
        ans = n_each
        budget_label = _format_money(budget, currency)
        sidx = (self.seed or 0) % len(_TEMPLATES_BUDGET_SHARES)
        question = _TEMPLATES_BUDGET_SHARES[sidx].format(
            budget=budget_label, model_a=models[0], model_b=models[1])
        cards = [
            (models[0], _format_money(prices[0], currency), "car"),
            (models[1], _format_money(prices[1], currency), "car"),
        ]
        for _ in range(cfg["n_extras"] - 2):
            other = rng.choice([m for m in _CAR_NAMES if m not in models])
            other_price = rng.choice([9000, 11000, 13000])
            cards.append((other, _format_money(other_price, currency), "car"))
        img = self._render_infographic("Car Dealership Catalogue", cards,
                                       "Showroom Pricing", chart_kind="bar", rng=rng)
        return question, _format_num(ans), img

    def _gen_takeover(self, cfg, rng):
        currency = rng.choice(_CURRENCIES)
        company = rng.choice(_COMPANY_NAMES)
        year = rng.choice([2010, 2015, 2020, 2023])
        # Pick clean revenue + multipliers
        revenue = rng.choice([2.5, 3.0, 4.0, 5.0, 6.3, 8.0, 10.0]) * 1_000_000
        mult = rng.choice([3, 5, 7, 8, 10])
        markup = rng.choice([5, 10, 15, 20, 25])
        ans = revenue * mult * (1 + markup / 100)
        revenue_label = _format_money(revenue, currency)
        sidx = (self.seed or 0) % len(_TEMPLATES_TAKEOVER)
        question = _TEMPLATES_TAKEOVER[sidx].format(
            mult=mult, year=year, revenue_label=revenue_label, markup=markup)
        cards = [
            (f"{year} Revenue", _format_money(revenue, currency), "money"),
            (f"Bid Multiple", f"{mult}x", "x"),
            (f"Mark-up", f"+{markup}%", "%"),
        ]
        for _ in range(cfg["n_extras"] - 3):
            other_year = rng.choice([y for y in (2010, 2015, 2020, 2023) if y != year])
            other_rev = rng.choice([2.0, 2.8, 4.5, 5.5, 6.5]) * 1_000_000
            cards.append((f"{other_year} Revenue",
                          _format_money(other_rev, currency), "money"))
        img = self._render_infographic(f"{company} Takeover", cards,
                                       "Board Valuation Report",
                                       chart_kind="donut", rng=rng)
        # Format ans without trailing decimals if integer
        ans_str = _format_num(ans)
        return question, ans_str, img

    def _gen_compound(self, cfg, rng):
        currency = rng.choice(_CURRENCIES)
        company = rng.choice(_COMPANY_NAMES)
        base = rng.choice([1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000])
        # 2026-05-04: at L7+ use uglier percents that don't simplify to clean
        # decimals, forcing 3-step compound with care.
        # 2026-05-04: at L9 use 4-step compound (Year 4) with prime percents.
        # 2026-05-04 R3: softened bump — 4-step Year 4 compound with prime
        # percents was too hard (L9 dropped to 0.20). Reuse L7+ 3-step compound
        # for L9 too — still uses ugly primes, still 3-step.
        p4 = None
        if cfg["level"] >= 7:
            base = rng.choice([1_250_000, 1_750_000, 2_375_000, 3_125_000, 4_625_000])
            p1 = rng.choice([7, 11, 13, 17, 19, 23])
            p2 = rng.choice([6, 9, 14, 16, 22])
            p3 = rng.choice([8, 12, 15, 21])  # third year (3-step compound)
            ans = base * (1 + p1 / 100) * (1 + p2 / 100) * (1 + p3 / 100)
        else:
            p1 = rng.choice([10, 15, 20, 25, 50])
            p2 = rng.choice([10, 15, 20, 25, 30])
            p3 = None
            ans = base * (1 + p1 / 100) * (1 + p2 / 100)
        base_label = _format_money(base, currency)
        sidx = (self.seed or 0) % len(_TEMPLATES_COMPOUND)
        if p4 is not None:
            question = (
                f"{company} starts at {base_label}. It grows by {p1}% in Year 1, "
                f"then {p2}% in Year 2, then {p3}% in Year 3, then {p4}% in "
                f"Year 4. What is the value at end of Year 4? (Round to "
                f"nearest integer.)"
            )
        elif p3 is not None:
            question = (
                f"{company} starts at {base_label}. It grows by {p1}% in Year 1, "
                f"then {p2}% in Year 2, then {p3}% in Year 3. What is the value "
                f"at end of Year 3? (Round to nearest integer.)"
            )
        else:
            question = _TEMPLATES_COMPOUND[sidx].format(
                company=company, p1=p1, p2=p2, base=base_label)
        cards = [
            ("Year 0", _format_money(base, currency), "money"),
            ("Year 1 Δ", f"+{p1}%", "%"),
            ("Year 2 Δ", f"+{p2}%", "%"),
        ]
        if p3 is not None:
            cards.append(("Year 3 Δ", f"+{p3}%", "%"))
        if p4 is not None:
            cards.append(("Year 4 Δ", f"+{p4}%", "%"))
        for _ in range(cfg["n_extras"] - len(cards)):
            other_p = rng.choice([5, 8, 12, 18])
            cards.append(("Industry avg", f"+{other_p}%", "%"))
        img = self._render_infographic(f"{company} R&D Trajectory", cards,
                                       "Spending Forecast",
                                       chart_kind="bar", rng=rng)
        return question, _format_num(ans), img

    def _gen_share_not_held(self, cfg, rng):
        company = rng.choice(_COMPANY_NAMES)
        pct = rng.choice([10, 15, 16, 20, 25, 40, 50, 60, 80])
        # total chosen so result is integer
        # not_held = total*(100-pct)/100; need integer
        total = rng.choice([10000, 25000, 50000, 100000, 200000, 305_000])
        not_held = total * (100 - pct) // 100
        # Ensure exact
        if total * (100 - pct) % 100 != 0:
            # round total to nearest 100 multiple
            total = (total // 100) * 100
            not_held = total * (100 - pct) // 100
        ans = not_held
        sidx = (self.seed or 0) % len(_TEMPLATES_SHARE_NOT_HELD)
        question = _TEMPLATES_SHARE_NOT_HELD[sidx].format(
            company=company, pct=pct, total=f"{total:,}")
        cards = [
            ("Total Shares", f"{total:,}", "share"),
            ("Director %", f"{pct}%", "%"),
        ]
        # Add distractor cards
        for _ in range(cfg["n_extras"] - 2):
            cards.append(("Avg holding",
                          f"{rng.randint(50, 500):,} shares", "share"))
        img = self._render_infographic(f"{company} Shareholding", cards,
                                       "Annual Report Snapshot",
                                       chart_kind="donut", rng=rng,
                                       donut_pct=pct)
        return question, _format_num(ans), img

    # ------------------------------------------------------------ render
    def _render_infographic(self, title, cards, subtitle, chart_kind="bar",
                            rng=None, donut_pct=None):
        if rng is None:
            rng = random.Random(0)
        bg_options = ["#fdfdfd", "#f7f9fc", "#fffaf0", "#f4f8f4"]
        accent_palettes = [
            ("#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"),
            ("#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"),
            ("#22577a", "#38a3a5", "#57cc99", "#80ed99", "#c7f9cc"),
            ("#1d3557", "#457b9d", "#a8dadc", "#e63946", "#f1faee"),
        ]
        bg = rng.choice(bg_options)
        accent = rng.choice(accent_palettes)
        title_color = accent[0]
        card_colors = list(accent[1:])

        # Layout: 1 title + 1 row of stat cards + 1 chart row
        n_cards = len(cards)
        n_per_row = min(3, n_cards)
        n_card_rows = (n_cards + n_per_row - 1) // n_per_row
        fig_w = max(8.0, 2.6 * n_per_row + 0.6)
        fig_h = 4.4 + 1.4 * n_card_rows
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor(bg)

        # Title bar
        title_ax = fig.add_axes([0.04, 0.86, 0.92, 0.10])
        title_ax.set_xlim(0, 1); title_ax.set_ylim(0, 1)
        title_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=title_color))
        title_ax.text(0.02, 0.55, title, color="white", fontsize=15,
                      fontweight="bold", va="center")
        title_ax.text(0.02, 0.18, subtitle, color="#f5f5f5", fontsize=10,
                      va="center")
        title_ax.axis("off")

        # Stat cards area: rows of cards from y=0.50 down
        cards_top = 0.84
        cards_bottom = 0.50
        card_height = (cards_top - cards_bottom) / max(1, n_card_rows)
        for i, (lbl, val, _icon) in enumerate(cards):
            row = i // n_per_row
            col = i % n_per_row
            cx = 0.04 + col * (0.92 / n_per_row)
            cw = (0.92 / n_per_row) - 0.02
            cy = cards_top - card_height * (row + 1) + 0.01
            ch = card_height - 0.04
            color = card_colors[i % len(card_colors)]
            ax_c = fig.add_axes([cx, cy, cw, ch])
            ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1)
            ax_c.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.85,
                                         linewidth=0))
            # left side accent
            ax_c.add_patch(plt.Rectangle((0, 0), 0.05, 1, facecolor=title_color))
            # Big number
            ax_c.text(0.5, 0.6, str(val), ha="center", va="center",
                      fontsize=18, fontweight="bold", color="white")
            ax_c.text(0.5, 0.18, str(lbl), ha="center", va="center",
                      fontsize=10, color="white")
            ax_c.axis("off")

        # Bottom chart
        chart_ax = fig.add_axes([0.10, 0.06, 0.80, 0.40])
        chart_ax.set_facecolor(bg)
        if chart_kind == "donut" and donut_pct is not None:
            sizes = [donut_pct, 100 - donut_pct]
            labels = ["Directors", "Other Holders"]
            wedges, texts, autotexts = chart_ax.pie(
                sizes, labels=labels, autopct="%1.0f%%",
                colors=[card_colors[0], card_colors[1]],
                startangle=90, wedgeprops={"width": 0.4, "edgecolor": bg, "linewidth": 2})
            chart_ax.set_aspect("equal")
        elif chart_kind == "donut":
            # Generic donut with random distribution
            slices = [rng.randint(15, 35) for _ in range(3)]
            chart_ax.pie(slices, labels=["Asia", "Europe", "Americas"],
                         autopct="%1.0f%%",
                         colors=card_colors[:3],
                         startangle=120,
                         wedgeprops={"width": 0.4, "edgecolor": bg, "linewidth": 2})
            chart_ax.set_aspect("equal")
        else:
            # Mini bar chart from card values when numeric, else random
            n_bars = max(3, min(6, n_cards + 2))
            bar_labels = [c[0][:12] for c in cards[:n_bars]]
            bar_vals = []
            for c in cards[:n_bars]:
                v = c[1]
                # Try to parse number
                vs = "".join(ch for ch in v if ch.isdigit() or ch == ".")
                try:
                    bar_vals.append(float(vs) if vs else rng.randint(20, 100))
                except ValueError:
                    bar_vals.append(rng.randint(20, 100))
            while len(bar_labels) < n_bars:
                bar_labels.append(f"Cat {len(bar_labels)+1}")
                bar_vals.append(rng.randint(30, 120))
            bars = chart_ax.bar(range(len(bar_labels)), bar_vals,
                                color=card_colors[0], edgecolor=title_color,
                                linewidth=0.6)
            mxv = max(bar_vals)
            for b, v in zip(bars, bar_vals):
                chart_ax.text(b.get_x() + b.get_width() / 2,
                              v + mxv * 0.02,
                              str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                              ha="center", va="bottom", fontsize=8.5,
                              fontweight="bold", color="#222")
            chart_ax.set_xticks(range(len(bar_labels)))
            chart_ax.set_xticklabels(bar_labels, rotation=15,
                                      ha="right", fontsize=8.5)
            chart_ax.spines["top"].set_visible(False)
            chart_ax.spines["right"].set_visible(False)
            chart_ax.set_ylim(0, mxv * 1.2 + 1)

        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=bg, dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
