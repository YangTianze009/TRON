"""
Hashi-style "bridges" puzzle.

2026-05-05 R5 P4: REWRITTEN with single-cell-style probe MCQ at L0-L4 to give
4B training a learnable signal. Full solve at L5-L9 constrained to small
3-island D=1 layouts (matching the benchmark D=1 sample style).

Rules (standard Hashi):
  - The grid contains "islands" at various cells; each island has a number
    1-8 indicating how many bridge endpoints leave it.
  - Bridges connect two islands and run horizontally OR vertically along
    grid rows/columns. They cannot turn.
  - Between any pair of islands you can have at most 2 bridges.
  - Bridges cannot cross other bridges (both horiz and vert).
  - All islands' degree-clue numbers must be exactly satisfied.
  - The whole island graph (with bridges) must be CONNECTED.

Per-level layout:
  L0/L1: 5x5 with 3 islands D=1; pre-draw partial bridges + ask one bridge
         pair count. 4-MCQ.
  L2/L3: D=1 5x5 still 3 islands but ask "how many bridges from this island
         total?" 4-MCQ.
  L4   : D=1 5x5; 5-MCQ with 15% trap rate.
  L5-L7: full-solve 5x5 D=1 (3 islands), text answer.
  L8/L9: full-solve as MCQ candidate-bridge-set + raised trap rate (30-40%).
"""
import random
import re
from typing import Dict, List, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ----------------------------------------------------------------------- #
# Probe MCQ templates (L0-L4)
# ----------------------------------------------------------------------- #
_PROBE_TEMPLATES = [
    "Look at the {n}x{n} bridges puzzle below.\n\n"
    "### Game Rules:\n"
    "1. Connect islands with bridges that run only horizontally or vertically.\n"
    "2. At most 2 bridges may connect any pair of islands.\n"
    "3. Bridges cannot cross any other bridge.\n"
    "4. The total bridge endpoints at each island must equal that island's number.\n"
    "5. All islands must form a single connected graph through the bridges.\n\n"
    "### Coordinate System:\n"
    "- Each island is at position (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "- Grid size: {n}x{n}\n"
    "- Islands (each entry is `(row, col): number`): {islands_text}\n\n"
    "### Question:\n"
    "{stem}\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",

    "{n}x{n} bridges puzzle.\n\n"
    "### Game Rules:\n"
    "- Bridges run only horizontally or vertically between two islands.\n"
    "- Up to 2 bridges per pair, no crossings.\n"
    "- Each island's number is the total count of bridge endpoints at that island.\n"
    "- All islands must be connected via the bridge network.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Islands: {islands_text}\n\n"
    "### Question:\n"
    "{stem}\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",

    "Bridges puzzle ({n}x{n}). Connect islands with horizontal/vertical bridges; max 2 per pair; no crossings; degrees match clues; the graph is connected.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Islands: {islands_text}\n\n"
    "### Question:\n"
    "{stem}\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",
]


# ----------------------------------------------------------------------- #
# Full-solve text templates (L5-L7)
# ----------------------------------------------------------------------- #
_TEMPLATES = [
    "Your task is to solve the {n}x{n} bridges puzzle described below.\n\n"
    "### Game Rules:\n"
    "1. Connect islands with horizontal/vertical bridges (max 2 per pair, no crossings).\n"
    "2. Each island's number is the total bridge endpoints at that island.\n"
    "3. All islands must form one connected graph.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "- Grid size: {n}x{n}\n"
    "- Islands: {islands_text}\n\n"
    "### Output Format:\n"
    "Output the bridges as space-separated `(r1,c1)-(r2,c2):n` entries (n is 1 or 2).\n"
    "Example: (0,0)-(0,4):1 (0,4)-(2,4):1",

    "Solve the bridges puzzle below.\n\n"
    "### Game Rules:\n"
    "Standard Hashi.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Islands: {islands_text}\n\n"
    "### Output Format:\n"
    "Output bridges as `(r1,c1)-(r2,c2):n` entries.",

    "Your task is to solve the bridges puzzle described below.\n\n"
    "### Game Rules:\n"
    "Standard Hashi: place horizontal/vertical bridges between islands; max 2 per pair; no crossings; bridge endpoints at each island sum to its clue number; the bridge graph must be fully connected.\n\n"
    "### Coordinate System:\n"
    "- {n}x{n} grid, 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "Islands and their bridge counts: {islands_text}\n\n"
    "### Output Format:\n"
    "Return the bridge list (each bridge as `(r1,c1)-(r2,c2):count`).",
]


def _gen_island_layout(n: int, num_islands: int,
                       rng: random.Random) -> Optional[List[Tuple[int, int]]]:
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells)
    chosen = []
    for cell in cells:
        ok = True
        for cr, cc in chosen:
            if abs(cr - cell[0]) <= 1 and abs(cc - cell[1]) <= 1:
                ok = False
                break
        if ok:
            chosen.append(cell)
        if len(chosen) >= num_islands:
            break
    if len(chosen) < num_islands:
        return None
    return chosen


def _bridges_could_collide(b1, b2):
    (a1r, a1c), (a2r, a2c) = b1
    (b1r, b1c), (b2r, b2c) = b2
    a_horiz = a1r == a2r
    b_horiz = b1r == b2r
    if a_horiz == b_horiz:
        return False
    if a_horiz:
        h_r = a1r
        h_c1, h_c2 = sorted([a1c, a2c])
        v_c = b1c
        v_r1, v_r2 = sorted([b1r, b2r])
    else:
        h_r = b1r
        h_c1, h_c2 = sorted([b1c, b2c])
        v_c = a1c
        v_r1, v_r2 = sorted([a1r, a2r])
    if v_c > h_c1 and v_c < h_c2 and h_r > v_r1 and h_r < v_r2:
        return True
    return False


def _possible_neighbors(island, islands):
    r, c = island
    isl_set = set(islands)
    max_r = max(rr for rr, _ in islands)
    max_c = max(cc for _, cc in islands)
    neighbors = []
    for rr in range(r - 1, -1, -1):
        if (rr, c) in isl_set:
            neighbors.append((rr, c))
            break
    for rr in range(r + 1, max_r + 1):
        if (rr, c) in isl_set:
            neighbors.append((rr, c))
            break
    for cc in range(c - 1, -1, -1):
        if (r, cc) in isl_set:
            neighbors.append((r, cc))
            break
    for cc in range(c + 1, max_c + 1):
        if (r, cc) in isl_set:
            neighbors.append((r, cc))
            break
    return neighbors


def _gen_solution(n, islands, rng):
    """Random valid bridge assignment."""
    edges = set()
    for isl in islands:
        for nb in _possible_neighbors(isl, islands):
            pair = tuple(sorted([isl, nb]))
            edges.add(pair)
    edges = list(edges)
    rng.shuffle(edges)

    bridges = {e: 0 for e in edges}
    deg = {isl: 0 for isl in islands}

    for _ in range(200):
        edges_shuffled = edges[:]
        rng.shuffle(edges_shuffled)
        changed = False
        for e in edges_shuffled:
            if bridges[e] >= 2:
                continue
            crosses = False
            for e2, cnt in bridges.items():
                if cnt > 0 and e2 != e and _bridges_could_collide(e, e2):
                    crosses = True
                    break
            if crosses:
                continue
            if rng.random() < 0.5:
                bridges[e] = bridges[e] + 1
                deg[e[0]] += 1
                deg[e[1]] += 1
                changed = True
        if not changed:
            break

    parent = {isl: isl for isl in islands}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for e, cnt in bridges.items():
        if cnt > 0:
            union(*e)
    roots = set(find(isl) for isl in islands)
    if len(roots) != 1:
        return None
    if any(d == 0 for d in deg.values()):
        return None
    if any(d > 8 for d in deg.values()):
        return None
    return {e: cnt for e, cnt in bridges.items() if cnt > 0}


def _validate_solution(islands, degrees, bridges):
    for e, cnt in bridges.items():
        if cnt not in (1, 2):
            return False
        if e[0] not in degrees or e[1] not in degrees:
            return False
        if e[0][0] != e[1][0] and e[0][1] != e[1][1]:
            return False
        possible = _possible_neighbors(e[0], islands)
        if e[1] not in possible:
            return False
    edges = list(bridges.keys())
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if _bridges_could_collide(edges[i], edges[j]):
                return False
    deg = {isl: 0 for isl in islands}
    for e, cnt in bridges.items():
        deg[e[0]] += cnt
        deg[e[1]] += cnt
    for isl, want in degrees.items():
        if deg.get(isl, 0) != want:
            return False
    parent = {isl: isl for isl in islands}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for e, cnt in bridges.items():
        if cnt > 0:
            union(*e)
    roots = set(find(isl) for isl in islands)
    return len(roots) == 1


def _solve_count(islands, degrees, limit=2):
    """Count valid bridge assignments matching the degrees, up to `limit`."""
    candidate_pairs = set()
    for isl in islands:
        for nb in _possible_neighbors(isl, islands):
            pair = tuple(sorted([isl, nb]))
            candidate_pairs.add(pair)
    edge_list = list(candidate_pairs)
    found = [0]

    def backtrack(idx, bridges, deg, used):
        if found[0] >= limit:
            return
        if idx == len(edge_list):
            # Check final
            for isl, want in degrees.items():
                if deg.get(isl, 0) != want:
                    return
            # Check connectivity
            parent = {isl: isl for isl in islands}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for (a, b), cnt in bridges.items():
                if cnt > 0:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
            roots = set(find(isl) for isl in islands)
            if len(roots) != 1:
                return
            found[0] += 1
            return
        e = edge_list[idx]
        for cnt in [0, 1, 2]:
            if cnt > 0 and any(_bridges_could_collide(e, e2) for e2 in used):
                continue
            new_deg = deg.copy()
            new_deg[e[0]] = new_deg.get(e[0], 0) + cnt
            new_deg[e[1]] = new_deg.get(e[1], 0) + cnt
            if new_deg[e[0]] > degrees[e[0]]:
                continue
            if new_deg[e[1]] > degrees[e[1]]:
                continue
            new_bridges = bridges.copy()
            new_bridges[e] = cnt
            new_used = used.copy()
            if cnt > 0:
                new_used.append(e)
            backtrack(idx + 1, new_bridges, new_deg, new_used)
            if found[0] >= limit:
                return

    backtrack(0, {}, {isl: 0 for isl in islands}, [])
    return found[0]


class HashiBridgesQA(StandaloneVisualEnv):
    ENV_NAME = "hashi_bridges"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0/L1: 5x5, 3 islands, single bridge-pair-count probe MCQ
        if level <= 1:
            return {"level": level, "n": 5, "num_islands": 3,
                    "mode": "probe_pair", "use_5_options": False,
                    "trap_rate": 0.0}
        # L2/L3: 5x5, 3 islands, "total degree" probe MCQ
        if level <= 3:
            return {"level": level, "n": 5, "num_islands": 3,
                    "mode": "probe_degree", "use_5_options": False,
                    "trap_rate": 0.0}
        # L4: 5x5, 3 islands, 5-MCQ + 15% trap
        if level == 4:
            return {"level": level, "n": 5, "num_islands": 3,
                    "mode": "probe_pair", "use_5_options": True,
                    "trap_rate": 0.15}
        # L5-L7: full-solve 5x5 D=1, text
        if level <= 7:
            return {"level": level, "n": 5, "num_islands": 3,
                    "mode": "full", "use_5_options": False,
                    "trap_rate": 0.0}
        # L8/L9: full-solve 5x5 D=1 as MCQ + raised trap
        return {"level": level, "n": 5, "num_islands": 3,
                "mode": "full", "use_5_options": True,
                "trap_rate": 0.40 if level == 9 else 0.30}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        num_islands = cfg["num_islands"]
        rng = random.Random((self.seed or 0) * 8093 + level * 53 + 17)
        mode = cfg["mode"]

        # Get a valid puzzle (random layout + valid bridge set)
        islands = None
        bridges = None
        degrees = None
        for attempt in range(120):
            islands_try = _gen_island_layout(n, num_islands, rng)
            if islands_try is None:
                continue
            sol = _gen_solution(n, islands_try, rng)
            if sol is None:
                continue
            degrees_try = {isl: 0 for isl in islands_try}
            for (a, b), cnt in sol.items():
                degrees_try[a] += cnt
                degrees_try[b] += cnt
            if not _validate_solution(islands_try, degrees_try, sol):
                continue
            # Need uniqueness for probe modes
            n_sols = _solve_count(islands_try, degrees_try, limit=2)
            if n_sols != 1:
                continue
            islands = islands_try
            bridges = sol
            degrees = degrees_try
            break
        if islands is None:
            return None

        self._n = n
        self._islands = islands
        self._degrees = degrees
        self._bridges = bridges
        self._mcq_mode = (mode != "full") or cfg.get("use_5_options", False)

        islands_text = ", ".join(f"({r},{c}):{degrees[(r, c)]}"
                                 for (r, c) in sorted(islands))

        if mode == "probe_pair":
            return self._build_probe_pair_question(
                cfg, rng, islands, degrees, bridges,
                islands_text, n, level,
            )
        elif mode == "probe_degree":
            return self._build_probe_degree_question(
                cfg, rng, islands, degrees, bridges,
                islands_text, n, level,
            )
        else:
            return self._build_full_question(
                cfg, rng, islands, degrees, bridges,
                islands_text, n, level,
            )

    # ------------------------------------------------------------------ #
    # Probe pair: "How many bridges between island X and island Y?"
    # The diagram pre-draws all bridges from the unique solution so the
    # model just visually counts the bridges between the queried pair.
    # ------------------------------------------------------------------ #
    def _build_probe_pair_question(self, cfg, rng, islands, degrees, bridges,
                                    islands_text, n, level):
        # Build set of candidate pairs (valid neighbors per Hashi rules)
        candidate_pairs = []
        for isl in islands:
            for nb in _possible_neighbors(isl, islands):
                pair = tuple(sorted([isl, nb]))
                candidate_pairs.append(pair)
        candidate_pairs = sorted(set(candidate_pairs))
        # All island pairs (also include disconnected pairs with answer 0)
        all_pairs = []
        for i in range(len(islands)):
            for j in range(i + 1, len(islands)):
                pair = tuple(sorted([islands[i], islands[j]]))
                all_pairs.append(pair)
        all_pairs = sorted(set(all_pairs))

        # Group by answer: each pair has answer = bridges.get(pair, 0).
        zero_pairs = [p for p in all_pairs if bridges.get(p, 0) == 0]
        one_pairs = [p for p in all_pairs if bridges.get(p, 0) == 1]
        two_pairs = [p for p in all_pairs if bridges.get(p, 0) == 2]
        buckets = []
        if zero_pairs:
            buckets.append(zero_pairs)
        if one_pairs:
            buckets.append(one_pairs)
        if two_pairs:
            buckets.append(two_pairs)
        if not buckets:
            return None
        bucket = rng.choice(buckets)
        target_pair = rng.choice(bucket)
        true_count = bridges.get(target_pair, 0)
        (r1, c1), (r2, c2) = target_pair
        stem = (
            f"In the diagram, how many bridges directly connect island "
            f"({r1},{c1}) with island ({r2},{c2})?"
        )

        # Options: 0, 1, 2 + "No correct answer" (4-MCQ); for 5-MCQ add
        # an extra plausible distractor count (3) or "Cannot be determined".
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        if use_5:
            base = ["0", "1", "2", "Cannot be determined"]
            # randomize order of base
            rng.shuffle(base)
            opt_texts = base + ["No correct answer"]
            true_text = str(true_count)
            use_trap = (cfg.get("trap_rate", 0.0) > 0
                        and rng.random() < cfg["trap_rate"])
            if use_trap:
                # Remove the correct numeric option from base, replace with
                # a different plausible distractor.
                opt_texts = [t for t in opt_texts[:4] if t != true_text]
                pad = ["3", "Cannot be determined", "Either 0 or 1"]
                rng.shuffle(pad)
                pi = 0
                while len(opt_texts) < 4:
                    if pad[pi] not in opt_texts:
                        opt_texts.append(pad[pi])
                    pi = (pi + 1) % len(pad)
                rng.shuffle(opt_texts)
                opt_texts.append("No correct answer")
                correct_letter = opt_letters[-1]
            else:
                # If true_text not in base (impossible since base contains 0/1/2),
                # find the letter
                if true_text not in opt_texts[:4]:
                    opt_texts[0] = true_text
                correct_letter = opt_letters[opt_texts.index(true_text)]
        else:
            base = ["0", "1", "2"]
            rng.shuffle(base)
            opt_texts = base + ["No correct answer"]
            true_text = str(true_count)
            correct_letter = opt_letters[opt_texts.index(true_text)]

        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
        question = _PROBE_TEMPLATES[sidx].format(
            n=n, islands_text=islands_text, stem=stem,
            options=opt_lines, letter_str=letter_str,
        )
        self._probe_target_pair = target_pair
        self._probe_count = true_count
        # Pre-draw all bridges so this is a visual-counting task
        img = self._render(n, islands, degrees, bridges_drawn=bridges,
                           highlight_pair=target_pair)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # Probe degree: "How many DISTINCT neighbors does this island connect to?"
    # The diagram pre-draws all bridges so it's a visual-counting task.
    # ------------------------------------------------------------------ #
    def _build_probe_degree_question(self, cfg, rng, islands, degrees, bridges,
                                      islands_text, n, level):
        # Pick an island and ask: how many DISTINCT neighbors does it have
        # bridges to?
        island = rng.choice(islands)
        ir, ic = island
        # Count distinct neighbors in `bridges` involving `island`
        distinct = 0
        for (a, b), cnt in bridges.items():
            if cnt > 0 and (a == island or b == island):
                distinct += 1
        # distinct ranges 1..3 for a 3-island puzzle (typical 1 or 2)
        true_count = distinct
        stem = (
            f"In the diagram, to how many DISTINCT other islands does "
            f"the highlighted island at ({ir},{ic}) connect via at least one bridge?"
        )

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        # Options: 1, 2, 3 + "No correct answer" (4-MCQ)
        # 5-MCQ: 0, 1, 2, 3 + "No correct answer"
        if use_5:
            base = ["0", "1", "2", "3"]
            rng.shuffle(base)
            opt_texts = base + ["No correct answer"]
            true_text = str(true_count)
            use_trap = (cfg.get("trap_rate", 0.0) > 0
                        and rng.random() < cfg["trap_rate"])
            if use_trap:
                opt_texts = [t for t in opt_texts[:4] if t != true_text]
                pad = ["4", "5", "Cannot be determined"]
                rng.shuffle(pad)
                pi = 0
                while len(opt_texts) < 4:
                    if pad[pi] not in opt_texts:
                        opt_texts.append(pad[pi])
                    pi = (pi + 1) % len(pad)
                rng.shuffle(opt_texts)
                opt_texts.append("No correct answer")
                correct_letter = opt_letters[-1]
            else:
                if true_text not in opt_texts[:4]:
                    opt_texts[0] = true_text
                correct_letter = opt_letters[opt_texts.index(true_text)]
        else:
            base = ["1", "2", "3"]
            rng.shuffle(base)
            opt_texts = base + ["No correct answer"]
            true_text = str(true_count)
            correct_letter = opt_letters[opt_texts.index(true_text)]

        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
        question = _PROBE_TEMPLATES[sidx].format(
            n=n, islands_text=islands_text, stem=stem,
            options=opt_lines, letter_str=letter_str,
        )
        self._probe_island = island
        self._probe_count = true_count
        img = self._render(n, islands, degrees, bridges_drawn=bridges,
                           highlight_island=island)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # Full mode (text or MCQ wrapper)
    # ------------------------------------------------------------------ #
    def _build_full_question(self, cfg, rng, islands, degrees, bridges,
                              islands_text, n, level):
        if cfg.get("use_5_options", False):
            return self._build_full_mcq_question(
                cfg, rng, islands, degrees, bridges,
                islands_text, n, level,
            )
        ans_str = " ".join(f"({a[0]},{a[1]})-({b[0]},{b[1]}):{cnt}"
                           for (a, b), cnt in sorted(bridges.items()))
        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(n=n, islands_text=islands_text)
        img = self._render(n, islands, degrees)
        self._mcq_mode = False
        return question, ans_str, img

    def _build_full_mcq_question(self, cfg, rng, islands, degrees, bridges,
                                  islands_text, n, level):
        true_bridges = sorted(bridges.items())
        true_str = " ".join(f"({a[0]},{a[1]})-({b[0]},{b[1]}):{cnt}"
                            for (a, b), cnt in true_bridges)

        # Build distractors by mutating: change a bridge count, swap counts,
        # add or drop a bridge.
        distractor_set = set()
        candidate_pairs = []
        for isl in islands:
            for nb in _possible_neighbors(isl, islands):
                pair = tuple(sorted([isl, nb]))
                candidate_pairs.append(pair)
        candidate_pairs = list(set(candidate_pairs))

        attempts = 0
        while len(distractor_set) < 6 and attempts < 200:
            attempts += 1
            new_bridges = dict(bridges)
            mutate_type = rng.choice(["change_count", "drop", "add"])
            if mutate_type == "change_count" and bridges:
                edges_list = list(new_bridges.keys())
                e = rng.choice(edges_list)
                new_cnt = 1 if new_bridges[e] == 2 else 2
                new_bridges[e] = new_cnt
            elif mutate_type == "drop" and bridges:
                edges_list = list(new_bridges.keys())
                e = rng.choice(edges_list)
                del new_bridges[e]
            elif mutate_type == "add":
                # Pick a candidate pair not in bridges
                avail = [p for p in candidate_pairs if p not in new_bridges]
                if not avail:
                    continue
                e = rng.choice(avail)
                new_bridges[e] = rng.choice([1, 2])
            new_repr = tuple(sorted((str(k), v) for k, v in new_bridges.items()))
            true_repr = tuple(sorted((str(k), v) for k, v in bridges.items()))
            if new_repr == true_repr:
                continue
            distractor_set.add(new_repr)

        if len(distractor_set) < 3:
            return None

        # Convert distractor reprs back to readable strings
        def _repr_to_str(repr_tuple):
            parts = []
            for kstr, v in repr_tuple:
                # k is e.g. "((0, 0), (0, 4))"
                # parse coords
                coords = re.findall(r"\d+", kstr)
                a_r, a_c, b_r, b_c = (int(x) for x in coords)
                parts.append(f"({a_r},{a_c})-({b_r},{b_c}):{v}")
            return " ".join(parts) if parts else "(no bridges)"

        distractors = list(distractor_set)
        rng.shuffle(distractors)

        use_5 = cfg["use_5_options"]
        n_distractor = 3 if use_5 else 2
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        use_trap = (cfg.get("trap_rate", 0.0) > 0
                    and rng.random() < cfg["trap_rate"])
        if use_trap and len(distractors) >= n_distractor + 1:
            options_data = distractors[: n_distractor + 1]
            options_data = options_data[: (4 if use_5 else 3)]
            rng.shuffle(options_data)
            correct_letter = opt_letters[-1]
        else:
            true_repr = tuple(sorted((str(k), v) for k, v in bridges.items()))
            options_data = [true_repr] + distractors[:n_distractor]
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(true_repr)]

        opt_texts = [_repr_to_str(d) for d in options_data]
        opt_texts.append("No correct answer")
        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        stem = (
            f"Look at the {n}x{n} bridges puzzle.\n\n"
            "### Game Rules:\n"
            "1. Bridges run only horizontally or vertically between two islands (max 2 per pair).\n"
            "2. Bridges cannot cross.\n"
            "3. The bridge endpoints at each island sum to its number.\n"
            "4. All islands form one connected graph.\n\n"
            "### Coordinate System:\n"
            "- (row, col), 0-indexed.\n\n"
            "### Current Puzzle State:\n"
            f"Islands: {islands_text}\n\n"
            "### Question:\n"
            "Which option lists the bridges in the unique solution?\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        img = self._render(n, islands, degrees)
        self._mcq_mode = True
        return stem, correct_letter, img

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, n, islands, degrees,
                bridges_drawn=None,
                highlight_pair=None,
                highlight_island=None) -> Image.Image:
        """Render the puzzle. If bridges_drawn is provided (probe modes),
        the bridges are drawn as lines between island centers — single
        bridge as one line, double bridge as two parallel lines.
        Optionally highlight one pair of islands (orange) or one island
        (orange ring) to direct the model's visual attention."""
        fig, ax = plt.subplots(figsize=(0.85 * (n + 1), 0.85 * (n + 1)), dpi=140)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Cell backgrounds
        for r in range(n):
            for c in range(n):
                ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                           facecolor="#fafafa", edgecolor="#ddd",
                                           linewidth=0.4))
        # Faint coord text on empty cells
        for r in range(n):
            for c in range(n):
                if (r, c) not in degrees:
                    ax.text(c + 0.06, n - 0.06 - r, f"{r},{c}",
                            ha="left", va="top", fontsize=6, color="#bbb")
        # Draw bridges FIRST (so islands cover their endpoints)
        if bridges_drawn:
            for (a, b), cnt in bridges_drawn.items():
                if cnt < 1:
                    continue
                (r1, c1), (r2, c2) = a, b
                # Convert (row, col) to plot (x, y): x=col+0.5, y=n-0.5-row
                x1, y1 = c1 + 0.5, n - 0.5 - r1
                x2, y2 = c2 + 0.5, n - 0.5 - r2
                # Color: highlight if matches highlight_pair
                pair = tuple(sorted([a, b]))
                is_hl = (highlight_pair is not None and pair == highlight_pair)
                color = "#cc6600" if is_hl else "#1d3557"
                lw = 3.5 if is_hl else 2.5
                if cnt == 1:
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
                            solid_capstyle="round", zorder=2)
                elif cnt == 2:
                    # Two parallel lines: offset perpendicular to the bridge.
                    # For a vertical line (c1 == c2), offset in x.
                    # For a horizontal line (r1 == r2), offset in y.
                    if c1 == c2:  # vertical
                        off = 0.10
                        ax.plot([x1 - off, x2 - off], [y1, y2],
                                color=color, linewidth=lw,
                                solid_capstyle="round", zorder=2)
                        ax.plot([x1 + off, x2 + off], [y1, y2],
                                color=color, linewidth=lw,
                                solid_capstyle="round", zorder=2)
                    else:  # horizontal
                        off = 0.10
                        ax.plot([x1, x2], [y1 - off, y2 - off],
                                color=color, linewidth=lw,
                                solid_capstyle="round", zorder=2)
                        ax.plot([x1, x2], [y1 + off, y2 + off],
                                color=color, linewidth=lw,
                                solid_capstyle="round", zorder=2)
        # Islands (drawn AFTER bridges so they cover endpoints)
        for (r, c), d in degrees.items():
            is_hl_island = (highlight_island is not None
                            and (r, c) == highlight_island)
            edge = "#cc6600" if is_hl_island else "#222"
            facec = "#fff5b1" if is_hl_island else "#ffe5b4"
            edge_lw = 3.0 if is_hl_island else 2.0
            ax.add_patch(plt.Circle((c + 0.5, n - 0.5 - r), 0.32,
                                     facecolor=facec, edgecolor=edge,
                                     linewidth=edge_lw, zorder=3))
            ax.text(c + 0.5, n - 0.5 - r, str(d),
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color="#222", zorder=4)
            ax.text(c + 0.5, n - 0.05 - r, f"({r},{c})",
                    ha="center", va="top", fontsize=7, color="#666",
                    zorder=4)
        ax.set_xlim(-0.05, n + 0.05)
        ax.set_ylim(-0.05, n + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ------------------------------------------------------------------ #
    # Answer checking — dispatches based on mode
    # ------------------------------------------------------------------ #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Fast-path guard: compute_score's fast-path sets _answer but skips
        # generate(), so _islands / _degrees / _mcq_mode etc. aren't set.
        # Raising AttributeError triggers reward_function.py's slow-path
        # fallback (re-run generate to populate state, then retry verify).
        if not hasattr(self, "_islands"):
            raise AttributeError("hashi state not initialized; need slow-path generate")
        if getattr(self, "_mcq_mode", False):
            return super()._check_answer(predicted, ground_truth)
        # Full-solve text mode
        edge_pat = re.compile(
            r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*-\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(\d+)"
        )
        bridges = {}
        for m in edge_pat.finditer(predicted):
            r1, c1, r2, c2, cnt = (int(x) for x in m.groups())
            a, b = (r1, c1), (r2, c2)
            if a == b:
                return False
            pair = tuple(sorted([a, b]))
            if pair in bridges:
                return False
            bridges[pair] = cnt
        if not bridges:
            return False
        return _validate_solution(self._islands, self._degrees, bridges)
