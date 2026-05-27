"""
Venn diagram QA -- 2 or 3 set Venn diagram with numbers in each region.
Questions: union size, intersection, only-in-A, symmetric difference.
"""
import math
import random
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context

_Q_UNION_2 = [
    "How many elements are in A union B?",
    "What is |A ∪ B| — the size of A union B?",
    "Count the elements in the union of sets A and B.",
    "How many elements belong to A or B (or both)?",
    "Report the total count of distinct elements across A ∪ B.",
    "Given the diagram, how many items are in A ∪ B?",
    "What is the size of A ∪ B based on the Venn diagram?",
    "How many elements lie in the union A ∪ B?",
    "Sum the elements in A or B (counted once). Report the total.",
    "How many distinct elements are in A or B together?",
    "Compute |A ∪ B| from the Venn diagram.",
    "What's the count of elements belonging to at least one of A, B?",
    "How many items are in A ∪ B in the diagram shown?",
    "Give the cardinality of A ∪ B.",
    "Based on the Venn diagram, how large is A ∪ B?",
    "Report the number of elements in the union A ∪ B.",
]

_Q_INTER_2 = [
    "How many elements are in A intersect B?",
    "What is |A ∩ B| in the diagram?",
    "Count the items that are in BOTH A and B.",
    "How many elements belong to both A and B?",
    "Report the size of A ∩ B.",
    "Give the cardinality of A ∩ B from the Venn diagram.",
    "How many elements lie in A and B (the overlap)?",
    "What is the count of the intersection A ∩ B?",
    "Based on the diagram, how many items are in A ∩ B?",
    "How many elements appear in both sets A and B simultaneously?",
    "Compute |A ∩ B| for the Venn diagram shown.",
    "How many elements are in the overlap region of A and B?",
    "Give |A ∩ B| from the diagram.",
    "Count the elements shared between A and B.",
    "What is the number of elements in A ∩ B?",
    "Report the count of A ∩ B based on the figure.",
]

_Q_ONLY_A_2 = [
    "How many elements are in A only (not in B)?",
    "Count the elements in A but not in B.",
    "What is |A \\ B| (elements of A excluding B)?",
    "How many items are in A and not in B?",
    "Based on the Venn diagram, how many elements are only in A?",
    "Report the count of elements exclusive to A.",
    "How many elements lie in A but not B?",
    "What is the size of A minus B?",
    "Give the number of elements that are in A alone.",
    "How many elements are unique to A (not shared with B)?",
    "Compute |A \\ B| from the diagram.",
    "What is the count of A-only elements (not in B)?",
    "In the Venn diagram, how many items are inside A but outside B?",
    "Report how many elements belong to A but not to B.",
    "Count the elements that are in A and not in B.",
    "What is the number of A-only elements (excluding any in B)?",
]

_Q_SYMDIFF_2 = [
    "What is the symmetric difference |A delta B| (in A or B but not both)?",
    "How many elements are in A or B but not both?",
    "Compute |A △ B| from the diagram (elements in exactly one of A, B).",
    "Give the size of the symmetric difference A △ B.",
    "How many elements belong to exactly one of A or B?",
    "What is the count of the symmetric difference of A and B?",
    "Count elements in A XOR B (in exactly one of the two sets).",
    "Report |A △ B| — elements in A or B but not their intersection.",
    "How many items are in A or B but excluded from A ∩ B?",
    "Sum of elements in (A \\ B) and (B \\ A). What is this total?",
    "Find the size of A △ B in the Venn diagram.",
    "What is |A △ B|, the count of exclusive-or elements?",
    "How many elements appear in exactly one of A, B (not both)?",
    "Compute the symmetric difference size for A and B.",
    "Report the number of elements that are in A or B but not both.",
    "Give the cardinality of A △ B based on the figure.",
]

_Q_TOTAL_2 = [
    "What is the total number of elements across all regions?",
    "Sum the counts in every region — what total do you get?",
    "Add the elements in all Venn regions. What is the sum?",
    "How many elements are there in total across the diagram?",
    "What is the overall element count across all regions?",
    "Report the total population of the Venn diagram.",
    "Count every element once across all regions. Report the total.",
    "What is the total cardinality of A ∪ B across all regions?",
    "Give the sum of all region counts in the Venn diagram.",
    "How many elements are depicted in total?",
    "What is the sum of all numbers shown in the Venn diagram?",
    "Report the grand total of elements across all regions.",
    "Based on the diagram, what is the total element count?",
    "Sum all region labels. What value is produced?",
    "What is the total number of distinct elements across the whole Venn diagram?",
    "Add up the entries in every Venn region; give the sum.",
]

_Q_COMPL_2 = [
    "Within the universe of A union B, how many elements are NOT in A (complement of A)?",
    "How many elements are in A ∪ B but NOT in A?",
    "Given the universe U = A ∪ B, what is |U \\ A|?",
    "Count the elements outside A in the universe A ∪ B.",
    "What is the size of the complement of A within A ∪ B?",
    "Report the number of elements in (A ∪ B) \\ A.",
    "How many elements of the union A ∪ B are outside A?",
    "Give |A^c| restricted to the universe A ∪ B.",
    "In the universe A ∪ B, how many elements are NOT in A?",
    "Count the elements of the universe that do not belong to A.",
    "What is |(A ∪ B) \\ A| in the diagram?",
    "How many non-A elements are in the universe A ∪ B?",
    "Report the number of non-A elements in the universe (A ∪ B).",
    "Give the count of elements in A ∪ B that lie outside A.",
    "What is the complement size of A in the universe A ∪ B?",
    "How many elements of A ∪ B fall outside of A?",
]

_Q_CONDPROB_2 = [
    "If an element is randomly chosen from set A, what is the probability it is also in B? Round to 2 decimals.",
    "What is P(B | A) in the Venn diagram? Round to 2 decimals.",
    "Choose an element uniformly from A. What is the chance it's also in B? Answer to 2 decimals.",
    "Compute the conditional probability P(B | A). Round to 2 decimals.",
    "What fraction of A's elements are also in B? Give 2 decimals.",
    "For a random element of A, what is P(in B | in A)? Round to 2 decimals.",
    "Based on the diagram, compute P(B | A) to 2 decimals.",
    "What is the probability an element of A is also in B? Round to 2 decimals.",
    "Given an element is in A, what's the probability it's also in B? Answer to 2 decimals.",
    "Calculate |A ∩ B| / |A| and round to 2 decimals.",
    "If we pick a random element from A, how likely is it to be in B? 2-decimal answer.",
    "What's P(B ∩ A) / P(A) from the diagram? Round to 2 decimals.",
    "Compute the conditional P(B | A) as a 2-decimal number.",
    "What proportion of A is shared with B? Round to 2 decimals.",
    "Given an element from A, what probability does it have of being in B? Round to 2 decimals.",
    "Report P(B | A) as a 2-decimal probability.",
]

_Q_UNION_3 = [
    "How many elements are in A union B union C?",
    "What is |A ∪ B ∪ C| in the diagram?",
    "How many elements belong to at least one of A, B, C?",
    "Report the size of A ∪ B ∪ C.",
    "Compute the cardinality of A ∪ B ∪ C.",
    "Count the distinct elements across A, B, C.",
    "Give |A ∪ B ∪ C| from the Venn diagram shown.",
    "How many items belong to A, B, or C (at least one)?",
    "Sum the populations of all regions. What is |A ∪ B ∪ C|?",
    "Based on the Venn diagram, what is the size of A ∪ B ∪ C?",
    "Count the total number of distinct elements in any of the three sets.",
    "How large is the union of the three sets shown?",
    "What is the total of A ∪ B ∪ C in this 3-set Venn?",
    "Give the count of elements in at least one of the three sets.",
    "Report |A ∪ B ∪ C| for the diagram.",
    "What is the cardinality of the union of A, B, and C?",
]

_Q_INTER_3 = [
    "How many elements are in A intersect B intersect C?",
    "What is |A ∩ B ∩ C|?",
    "Count the elements that lie in all three of A, B, C.",
    "Give the size of A ∩ B ∩ C from the Venn diagram.",
    "How many items are shared by all three sets?",
    "Report |A ∩ B ∩ C| in the diagram.",
    "How many elements belong to A, B, AND C simultaneously?",
    "Compute the count of A ∩ B ∩ C.",
    "What is the cardinality of the triple intersection?",
    "Based on the Venn, how many elements are in all three sets?",
    "How many common elements do A, B, and C share?",
    "Count the elements in the center triple-overlap region.",
    "State the size of the triple intersection A ∩ B ∩ C.",
    "What's |A ∩ B ∩ C| in the figure?",
    "Give the number of elements in all three sets A, B, C.",
    "How many items lie in the intersection of A, B, and C?",
]

_Q_ONLY_A_3 = [
    "How many elements are only in A (not in B or C)?",
    "Count the elements exclusive to A (not in B, not in C).",
    "Report the size of A \\ (B ∪ C).",
    "What is |A only| — elements in A but not in B or C?",
    "How many items are in A and neither in B nor in C?",
    "Compute |A \\ (B ∪ C)| from the diagram.",
    "Count the elements that are ONLY in A.",
    "What is the number of elements exclusive to A?",
    "How many items are in A but outside both B and C?",
    "Give the count of A-only elements (no membership in B or C).",
    "How many elements lie in A but not B and not C?",
    "What is the A-only region's cardinality?",
    "Count items in A that are excluded from both B and C.",
    "Report how many elements are in A alone (not shared with B or C).",
    "Find the size of the A-only region of the Venn diagram.",
    "How many elements belong to A only (not B, not C)?",
]

_Q_SYMDIFF_3 = [
    "How many elements are in A or B but not both (ignoring C membership)?",
    "Count elements in A △ B (symmetric difference, ignoring C).",
    "How many items are in exactly one of A, B (without caring about C)?",
    "Give the size of A △ B (A XOR B) ignoring C.",
    "Compute |A △ B| ignoring C membership.",
    "How many elements are in A or B but not their intersection (ignoring C)?",
    "Report the symmetric difference count of A and B, independent of C.",
    "How many elements belong to exactly one of A, B?",
    "Find the count of A XOR B regardless of C.",
    "What is |A △ B| when we ignore C in the Venn?",
    "Count the elements in A or B but not in A ∩ B.",
    "How many items lie in (A ∪ B) \\ (A ∩ B)?",
    "Report the number of elements in the symmetric difference A △ B.",
    "What is the cardinality of A XOR B ignoring C?",
    "How many items belong to A or B but not both (C unconstrained)?",
    "Give the size of the symmetric difference of A and B.",
]

_Q_TOTAL_3 = [
    "What is the total across all regions of the Venn diagram?",
    "Sum the counts in every region. What total do you get?",
    "Report the total element count across all regions of the 3-set Venn.",
    "How many elements are there in total across all seven regions?",
    "What is the grand total for the Venn diagram?",
    "Add up the numbers across every Venn region. What is the sum?",
    "Give |A ∪ B ∪ C| — the total count across all regions.",
    "How many items are shown in total in the Venn diagram?",
    "Sum all region labels in the diagram. Report the total.",
    "What is the overall element count in the 3-set Venn?",
    "Based on the diagram, what is the total population?",
    "Count every element once. What is the grand total?",
    "What is the total cardinality of A ∪ B ∪ C across all regions?",
    "Sum the numbers in all regions of the Venn diagram.",
    "Give the total element count across the whole diagram.",
    "How many distinct elements are depicted across the 3 sets?",
]

_Q_COMPL_3 = [
    "Within the universe of all sets, how many elements are NOT in A?",
    "How many elements of A ∪ B ∪ C are NOT in A?",
    "Compute |(A ∪ B ∪ C) \\ A| for this Venn diagram.",
    "Count the elements outside A in the universe A ∪ B ∪ C.",
    "What is the size of A's complement within A ∪ B ∪ C?",
    "Report the number of non-A elements in the universe.",
    "How many elements are in the universe but not in A?",
    "Give the count of elements NOT in A in the diagram.",
    "Find the size of (A ∪ B ∪ C) \\ A.",
    "What is |A^c| restricted to A ∪ B ∪ C?",
    "How many elements lie outside A but inside the Venn universe?",
    "Based on the diagram, how many elements are NOT in set A?",
    "Count items in the 3-set universe that do not belong to A.",
    "Give the cardinality of the complement of A within A ∪ B ∪ C.",
    "What is the count of elements in A ∪ B ∪ C but not in A?",
    "Report how many elements are outside A in the Venn universe.",
]

_Q_CONDPROB_3 = [
    "If an element is randomly chosen from set A, what is the probability it is also in B? Round to 2 decimals.",
    "What is P(B | A) in the 3-set Venn diagram? Round to 2 decimals.",
    "Picking an element uniformly from A, what is the chance it's in B? 2-decimal answer.",
    "Compute P(B | A) = |A ∩ B| / |A|. Round to 2 decimals.",
    "What fraction of A's elements are also in B? Round to 2 decimals.",
    "For a random element of A, what is P(in B | in A)? 2 decimals.",
    "Compute the conditional probability P(B | A) from the Venn diagram. Round to 2 decimals.",
    "Given an element of A, what is the probability it is also in B? 2-decimal answer.",
    "Calculate |A ∩ B| / |A| and round to 2 decimals.",
    "What is P(B | A) based on the Venn diagram? Answer to 2 decimals.",
    "Find the conditional probability that a random element of A also belongs to B. Round to 2 decimals.",
    "What proportion of A's elements are shared with B? Round to 2 decimals.",
    "Given it's in A, what is the probability it's in B? 2-decimal answer.",
    "Report P(B | A) to 2 decimals based on the diagram.",
    "If we sample uniformly from A, what is the probability of hitting B? Round to 2 decimals.",
    "What is the P(B | A)? Round to 2 decimal places.",
]

class VennDiagramQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "venn_diagram"

    def _level_config(self, level: int) -> dict:
        if level <= 1:
            return {"qtypes": ["union", "intersection", "only_a"], "n_sets": 2}
        if level <= 3:
            return {"qtypes": ["union", "intersection", "only_a", "total"], "n_sets": 2}
        if level <= 5:
            return {"qtypes": ["union", "intersection", "only_a", "symmetric_diff", "total"],
                    "n_sets": None}  # random 2 or 3
        if level <= 7:
            return {"qtypes": ["union", "intersection", "only_a", "symmetric_diff",
                               "complement", "conditional_probability"], "n_sets": 3}
        return {"qtypes": ["union", "intersection", "only_a", "symmetric_diff",
                           "complement", "conditional_probability", "total"],
                "n_sets": 3}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        style = self._random_style()
        n_sets_cfg = cfg.get("n_sets")
        n_sets = parameter.get("n_sets", n_sets_cfg if n_sets_cfg else rng.choice([2, 2, 3]))
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        if n_sets == 2:
            return self._two_set(rng, sub_rng, style, qtype)
        else:
            return self._three_set(rng, sub_rng, style, qtype)

    def _pick_mode_params(self, rng, style):
        mode = pick_render_mode(rng)
        if mode == "textbook":
            tbp = textbook_params(rng)
            bg = tbp["bg"]
            edge = tbp["line_color"]
            dpi = tbp["dpi"]
            font_kw = {"fontfamily": tbp["font_family"]}
        elif mode == "sketch":
            bg = rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            edge = "#1a1a1a"
            dpi = style["dpi"]
            font_kw = {}
        else:
            bg = style["bg_color"]
            edge = style["geo_line_color"]
            dpi = style["dpi"]
            font_kw = {}
        return mode, bg, edge, dpi, font_kw

    def _two_set(self, rng, sub_rng, style, qtype):
        only_a = rng.randint(3, 20)
        only_b = rng.randint(3, 20)
        ab = rng.randint(1, 12)

        sc = style["figsize_scale"]
        mode, bg, edge, dpi, font_kw = self._pick_mode_params(sub_rng, style)

        # Random layout variation
        jitter = sub_rng.uniform(-0.15, 0.15)
        radius = sub_rng.uniform(1.3, 1.65)
        sep = sub_rng.uniform(0.5, 0.8)

        palette_perm = list(style["palette"])
        sub_rng.shuffle(palette_perm)

        def _draw():
            fig, ax = plt.subplots(figsize=(7 * sc, 5 * sc))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            c1 = plt.Circle((-sep, jitter), radius, alpha=style["geo_fill_alpha"],
                            fc=palette_perm[0], ec=edge, lw=style["line_width"])
            c2 = plt.Circle((sep, -jitter), radius, alpha=style["geo_fill_alpha"],
                            fc=palette_perm[1], ec=edge, lw=style["line_width"])
            ax.add_patch(c1); ax.add_patch(c2)
            fs = style["font_size_base"] + 2
            ax.text(-radius * 0.85, jitter, str(only_a), fontsize=fs, fontweight="bold",
                    ha="center", va="center", **font_kw)
            ax.text(0, (jitter - jitter) / 2, str(ab), fontsize=fs, fontweight="bold",
                    ha="center", va="center", **font_kw)
            ax.text(radius * 0.85, -jitter, str(only_b), fontsize=fs, fontweight="bold",
                    ha="center", va="center", **font_kw)
            ax.text(-sep, radius + 0.2, "A", fontsize=fs + 2, fontweight="bold",
                    ha="center", **font_kw)
            ax.text(sep, radius + 0.2, "B", fontsize=fs + 2, fontweight="bold",
                    ha="center", **font_kw)
            ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.1, 2.4); ax.set_aspect("equal"); ax.axis("off")
            ax.set_title("Venn Diagram", fontsize=style["font_size_base"] + 3,
                         fontweight="bold", **font_kw)
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=70, randomness=1.3):
                fig = _draw()
        else:
            fig = _draw()
        img = self.fig_to_pil(fig, dpi=dpi)

        union = only_a + only_b + ab
        sym_diff = only_a + only_b
        sidx = (self.seed or 0) % 16
        if qtype == "union":
            return _Q_UNION_2[sidx], str(union), img
        elif qtype == "intersection":
            return _Q_INTER_2[sidx], str(ab), img
        elif qtype == "only_a":
            return _Q_ONLY_A_2[sidx], str(only_a), img
        elif qtype == "symmetric_diff":
            return _Q_SYMDIFF_2[sidx], str(sym_diff), img
        elif qtype == "total":
            return _Q_TOTAL_2[sidx], str(union), img
        elif qtype == "complement":
            comp_a = only_b
            return _Q_COMPL_2[sidx], str(comp_a), img
        elif qtype == "conditional_probability":
            total_a = only_a + ab
            if total_a == 0:
                return None
            prob = round(ab / total_a, 2)
            return _Q_CONDPROB_2[sidx], str(prob), img
        return None

    def _three_set(self, rng, sub_rng, style, qtype):
        only_a = rng.randint(2, 15)
        only_b = rng.randint(2, 15)
        only_c = rng.randint(2, 15)
        ab = rng.randint(1, 8)
        ac = rng.randint(1, 8)
        bc = rng.randint(1, 8)
        abc = rng.randint(0, 5)

        sc = style["figsize_scale"]
        mode, bg, edge, dpi, font_kw = self._pick_mode_params(sub_rng, style)

        radius = sub_rng.uniform(1.15, 1.45)
        x_offset = sub_rng.uniform(0.4, 0.6)
        y_offset = sub_rng.uniform(0.3, 0.5)

        palette_perm = list(style["palette"])
        sub_rng.shuffle(palette_perm)

        def _draw():
            fig, ax = plt.subplots(figsize=(7 * sc, 6 * sc))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            centers = [(-x_offset, y_offset), (x_offset, y_offset), (0, -y_offset - 0.1)]
            names = ["A", "B", "C"]
            for i, (cx, cy) in enumerate(centers):
                c = plt.Circle((cx, cy), radius, alpha=style["geo_fill_alpha"],
                               fc=palette_perm[i], ec=edge, lw=style["line_width"])
                ax.add_patch(c)
                ax.text(cx + (cx - 0) * 0.8, cy + (cy - 0) * 0.8 + 0.3, names[i],
                        fontsize=style["font_size_base"] + 3, fontweight="bold",
                        ha="center", **font_kw)

            fs = style["font_size_base"] + 1
            ax.text(-x_offset - radius * 0.65, y_offset + 0.2, str(only_a),
                    fontsize=fs, fontweight="bold", ha="center", **font_kw)
            ax.text(x_offset + radius * 0.65, y_offset + 0.2, str(only_b),
                    fontsize=fs, fontweight="bold", ha="center", **font_kw)
            ax.text(0, -y_offset - radius * 0.65, str(only_c),
                    fontsize=fs, fontweight="bold", ha="center", **font_kw)
            ax.text(0, y_offset + 0.4, str(ab), fontsize=fs,
                    fontweight="bold", ha="center", **font_kw)
            ax.text(-x_offset - 0.05, -0.15, str(ac), fontsize=fs,
                    fontweight="bold", ha="center", **font_kw)
            ax.text(x_offset + 0.05, -0.15, str(bc), fontsize=fs,
                    fontweight="bold", ha="center", **font_kw)
            ax.text(0, 0.1, str(abc), fontsize=fs,
                    fontweight="bold", ha="center", color="red", **font_kw)
            ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.4, 2.5); ax.set_aspect("equal"); ax.axis("off")
            ax.set_title("Venn Diagram (3 Sets)", fontsize=style["font_size_base"] + 3,
                         fontweight="bold", **font_kw)
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=70, randomness=1.3):
                fig = _draw()
        else:
            fig = _draw()
        img = self.fig_to_pil(fig, dpi=dpi)

        union = only_a + only_b + only_c + ab + ac + bc + abc
        intersect_all = abc
        sidx = (self.seed or 0) % 16
        if qtype == "union":
            return _Q_UNION_3[sidx], str(union), img
        elif qtype == "intersection":
            return _Q_INTER_3[sidx], str(intersect_all), img
        elif qtype == "only_a":
            return _Q_ONLY_A_3[sidx], str(only_a), img
        elif qtype == "symmetric_diff":
            sd = only_a + only_b + ac + bc
            return _Q_SYMDIFF_3[sidx], str(sd), img
        elif qtype == "total":
            return _Q_TOTAL_3[sidx], str(union), img
        elif qtype == "complement":
            comp_a = only_b + only_c + bc
            return _Q_COMPL_3[sidx], str(comp_a), img
        elif qtype == "conditional_probability":
            total_a = only_a + ab + ac + abc
            if total_a == 0:
                return None
            prob = round((ab + abc) / total_a, 2)
            return _Q_CONDPROB_3[sidx], str(prob), img
        return None
