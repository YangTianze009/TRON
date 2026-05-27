"""
Word Ladder: given a start word and a target word of the same length, find a
sequence of valid English words where each adjacent pair differs by exactly
one letter. The full path including endpoints must use only the provided
dictionary (rendered in the image).

Difficulty axes:
  - word length (3 → 5)
  - path length (3 → 5)
"""
import random
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the WordLadder puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Transform the start word into the target word, one letter change per step.\n"
    "2. Each intermediate word must come from the provided dictionary.\n"
    "3. Each consecutive pair of words must differ in exactly one letter (same length).\n\n"
    "### Coordinate System:\n"
    "- Words are lowercase English; word length is fixed.\n\n"
    "### Current Puzzle State:\n"
    "- Start word: {start}\n"
    "- Target word: {target}\n"
    "- Dictionary: {dictionary}\n\n"
    "### Output Format:\n"
    "Provide the path as a Python list of word strings inside <answer>...</answer>.\n"
    "Example: <answer>[\"til\", \"tit\", \"tat\", \"eat\"]</answer>",

    "Solve the WordLadder puzzle below.\n\n"
    "### Game Rules:\n"
    "- Begin at the start word and end at the target word.\n"
    "- Each step changes exactly one letter, keeping the word length constant.\n"
    "- Every word in the chain (including endpoints) must appear in the dictionary.\n\n"
    "### Coordinate System:\n"
    "- Words are strings of equal length.\n\n"
    "### Current Puzzle State:\n"
    "Start: {start}\n"
    "Target: {target}\n"
    "Dictionary: {dictionary}\n\n"
    "### Output Format:\n"
    "Output the chain as a comma-separated list of words inside <answer>...</answer>.",

    "Your task is to find a WordLadder from start to target.\n\n"
    "### Game Rules:\n"
    "Each step in the ladder changes exactly one letter; every word must be in the dictionary.\n\n"
    "### Coordinate System:\n"
    "- Words are equal-length lowercase strings.\n\n"
    "### Current Puzzle State:\n"
    "start = {start}\n"
    "target = {target}\n"
    "dictionary = {dictionary}\n\n"
    "### Output Format:\n"
    "Output the path inside <answer>...</answer> as a Python list of strings.",
]

# Small built-in word lists by length for puzzle generation. Picked common words
# with rich one-letter-change connectivity.
_WORDS_3 = [
    "cat", "bat", "rat", "hat", "mat", "pat", "sat", "vat", "fat", "tat",
    "cot", "bot", "rot", "hot", "got", "pot", "lot", "dot", "not", "tot",
    "cap", "tap", "lap", "map", "sap", "rap", "nap", "gap", "zap",
    "bag", "rag", "tag", "wag", "lag", "sag", "nag",
    "dog", "log", "bog", "fog", "hog", "jog",
    "tin", "bin", "fin", "win", "din", "pin", "sin", "kin",
    "ten", "men", "den", "hen", "pen", "yen",
    "tan", "ban", "can", "fan", "man", "pan", "ran", "van", "wan", "gan",
    "tip", "dip", "hip", "lip", "nip", "pip", "rip", "sip", "zip",
    "top", "cop", "hop", "lop", "mop", "pop", "sop", "fop",
    "but", "cut", "gut", "hut", "jut", "nut", "put", "rut",
    "bit", "fit", "hit", "kit", "lit", "pit", "sit", "wit",
    "bee", "fee", "see", "tee", "wee", "lee",
    "bay", "day", "gay", "hay", "jay", "lay", "may", "pay", "ray", "say", "way",
    "bed", "fed", "led", "red", "wed",
    "big", "dig", "fig", "gig", "jig", "pig", "wig", "rig",
    "boy", "coy", "joy", "soy", "toy",
    "set", "bet", "get", "jet", "let", "met", "net", "pet", "vet", "wet", "yet",
    "bow", "cow", "how", "low", "mow", "now", "row", "sow", "tow", "wow",
    "but", "cup", "pup", "sup",
]

_WORDS_4 = [
    "cold", "cord", "card", "ward", "warm", "worm", "word", "wore", "core",
    "cone", "code", "cope", "rope", "ripe", "wipe", "pipe",
    "best", "rest", "test", "vest", "west", "lest", "nest", "pest",
    "mile", "mild", "wild", "wile", "wile",
    "dare", "bare", "care", "fare", "hare", "mare", "pare", "rare", "tare",
    "born", "burn", "barn", "darn", "yarn", "warn", "horn",
    "hand", "hard", "hare", "ware", "were",
    "pale", "tale", "bale", "gale", "hale", "kale", "male", "sale", "vale",
    "stop", "step", "seep", "deep", "deed", "feed", "feet", "feel", "heel", "heap",
    "town", "tort", "tone", "torn", "term", "tern", "ten",
    "land", "lamp", "lump", "limp", "lime", "lite", "life",
    "make", "wake", "lake", "rake", "sake", "take", "bake", "cake", "fake", "jake",
    "love", "live", "give", "dive", "five", "hive", "rive", "wive",
    "dawn", "down", "town", "torn", "tone",
    "hike", "bike", "like", "mike", "pike", "sike", "tike",
    "cool", "fool", "pool", "tool", "wool", "boom", "doom", "loom", "room", "zoom",
    "mind", "mend", "send", "send", "tend", "vend", "wend", "fend",
    "trip", "trim", "drip", "grip",
    "love", "lobe", "lone", "lore",
    "ring", "rink", "sink", "wink", "pink", "link", "kink", "mink",
]

_WORDS_BY_LEN = {3: _WORDS_3, 4: _WORDS_4}


def _diff_one(w1: str, w2: str) -> bool:
    if len(w1) != len(w2):
        return False
    diff = sum(1 for a, b in zip(w1, w2) if a != b)
    return diff == 1


def _bfs_path(start: str, target: str, dictionary: Set[str]) -> Optional[List[str]]:
    from collections import deque
    if start == target:
        return [start]
    if start not in dictionary or target not in dictionary:
        return None
    parent = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in dictionary:
            if v in parent:
                continue
            if _diff_one(u, v):
                parent[v] = u
                if v == target:
                    path = [v]
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    return list(reversed(path))
                q.append(v)
    return None


class WordLadderQA(StandaloneVisualEnv):
    ENV_NAME = "word_ladder"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        word_len = 3 if level <= 4 else 4
        target_path_len = 3 + (level // 3)  # 3 → 6
        return {"level": level, "word_len": word_len,
                "target_path_len": target_path_len}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1093 + level * 97 + 29)
        wl = cfg["word_len"]
        words = _WORDS_BY_LEN.get(wl, _WORDS_3)
        # Pick a start, find a target reachable in target_path_len steps
        for _ in range(50):
            start = rng.choice(words)
            # BFS from start to find reachable words and their distances
            from collections import deque
            dist = {start: 0}
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= cfg["target_path_len"] + 2:
                    continue
                for v in words:
                    if v in dist:
                        continue
                    if _diff_one(u, v):
                        dist[v] = dist[u] + 1
                        q.append(v)
            # Pick target with desired distance
            candidates = [w for w, d in dist.items()
                          if d == cfg["target_path_len"] - 1]
            if not candidates:
                continue
            target = rng.choice(candidates)
            # Build a small dictionary subset that contains the path + decoys
            full_dict_set = set(words)
            path = _bfs_path(start, target, full_dict_set)
            if path is None or len(path) != cfg["target_path_len"]:
                continue
            # Dictionary: path + a few decoys
            decoys = rng.sample(
                [w for w in words if w not in path],
                min(8, max(0, len(words) - len(path)))
            )
            dictionary = list(set(path + decoys))
            rng.shuffle(dictionary)
            ans_str = ", ".join(path)

            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx].format(
                start=start, target=target, dictionary=sorted(dictionary),
            )
            self._dict = set(dictionary)
            self._start = start
            self._target = target
            img = self._render(start, target, dictionary)
            return question, ans_str, img
        return None

    def _render(self, start, target, dictionary) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.text(0.05, 0.93, "Start:", fontsize=12, color="#2c3e50",
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.20, 0.93, start.upper(), fontsize=16, color="#1a5e1a",
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.45, 0.93, "Target:", fontsize=12, color="#2c3e50",
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.62, 0.93, target.upper(), fontsize=16, color="#a52a2a",
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.83, "Dictionary:", fontsize=12, color="#2c3e50",
                fontweight="bold", transform=ax.transAxes)
        # Render dictionary as grid
        per_row = 4
        for i, w in enumerate(dictionary):
            r, c = divmod(i, per_row)
            ax.text(0.08 + c * 0.22, 0.75 - r * 0.10, w.upper(),
                    fontsize=13, color="#2c3e50", transform=ax.transAxes,
                    fontfamily="monospace")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Parse predicted as Python list-of-strings or comma-separated word list
        import re, ast
        s = predicted.strip()
        s = re.sub(r"```[^\n]*\n", "", s).replace("```", "").strip()
        toks = None
        # Try Python literal list first
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)) and all(isinstance(x, str) for x in obj):
                toks = [str(x).strip().lower() for x in obj]
        except (ValueError, SyntaxError, TypeError):
            pass
        if toks is None:
            # Strip brackets / quotes, split on comma/whitespace
            cleaned = re.sub(r'[\[\]"\']', " ", s)
            toks = [t.strip().lower() for t in re.split(r"[,\s]+", cleaned)
                    if t.strip()]
        if not toks or len(toks) < 2:
            return False
        if toks[0] != self._start or toks[-1] != self._target:
            return False
        for w in toks:
            if w not in self._dict:
                return False
        for a, b in zip(toks, toks[1:]):
            if not _diff_one(a, b):
                return False
        return True
