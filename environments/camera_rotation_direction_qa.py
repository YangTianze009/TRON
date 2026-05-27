"""
Camera Rotation Direction QA (v5 G14b, for Motion-Cam (spatial)).

2026-04-26 REDESIGN: dropped matplotlib 3D azim/elev rendering (too subtle —
4B/8B VL models can't read perspective shifts). Replaced with a custom 2D
panoramic first-person view: landmarks live at fixed world-(yaw, pitch)
positions; the "photo" is a window cut from the panorama at the camera's
current orientation. Horizontal/vertical degree markers at the panel edge
let the model read the angular position of each landmark directly.

Visual mapping (the only thing the model has to learn):
  - Camera turns RIGHT  → landmarks shift LEFT  in the photo
  - Camera turns LEFT   → landmarks shift RIGHT in the photo
  - Camera tilts UP     → landmarks shift DOWN  in the photo
  - Camera tilts DOWN   → landmarks shift UP    in the photo

Reward: MCQ letter (A=left, B=right, C=up, D=down).

Level axes:
  A) Rotation magnitude:   80° at L0/L1, 50° at L2/L3, 30° at L4-6, 15° at L7+
  B) Number of landmarks:  3 at L0, up to 6 at L9
  C) Distractor noise:     none at L0, small jitter at L4+
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "Two first-person photos taken from the SAME spot — only the camera direction changed. In which direction did the camera rotate from photo 1 → photo 2? A. turned left  B. turned right  C. tilted up  D. tilted down. Put the letter in <answer>...</answer>.",
    "Camera fixed in place, orientation rotated between photo 1 and photo 2. Identify the rotation direction. A. left  B. right  C. up  D. down. Put letter in <answer>...</answer>.",
    "Same position, different camera angle. From photo 1 to photo 2, the camera rotated: A. left  B. right  C. up  D. down. Put letter in <answer>...</answer>.",
    "Two views of the same scene from the same point. Direction of camera rotation (1 → 2)? A-D = left/right/up/down. Put letter in <answer>...</answer>.",
    "First-person pair (no translation, only rotation). Rotation direction? A-D. Put letter in <answer>...</answer>.",
    "Identify the camera rotation between the two photos. A. turned left  B. turned right  C. tilted up  D. tilted down. Put letter in <answer>...</answer>.",
    "How did the camera rotate between photo 1 and photo 2? A-D. Put letter in <answer>...</answer>.",
    "Photo 1 → Photo 2: camera rotated in which direction? A-D. Put letter in <answer>...</answer>.",
    "Camera rotation (no movement) between these views — which direction? A-D. Put letter in <answer>...</answer>.",
    "Determine the camera rotation direction. A-D. Put letter in <answer>...</answer>.",
    "Between photo 1 and photo 2, the camera turned: A-D. Put letter in <answer>...</answer>.",
    "Ego-rotation direction between the two views? A-D. Put letter in <answer>...</answer>.",
    "Rotation direction (not translation). A-D. Put letter in <answer>...</answer>.",
    "Two photos, camera at same point but rotated differently. Direction of rotation? A-D. Put letter in <answer>...</answer>.",
    "Identify ego-rotation from photo 1 to photo 2. A-D. Put letter in <answer>...</answer>.",
    "Camera rotated (no translation). Rotation direction? A-D. Put letter in <answer>...</answer>.",
]

# Color palette for landmarks (color name → hex)
_LANDMARK_COLORS = [
    ("red",     "#e74c3c"),
    ("green",   "#27ae60"),
    ("blue",    "#3498db"),
    ("orange",  "#f39c12"),
    ("purple",  "#9b59b6"),
    ("teal",    "#1abc9c"),
]


def _render_pano_view(landmarks, cam_yaw, cam_pitch,
                      fov_h=120, fov_v=80, ax=None, title=""):
    """
    Render a 2D first-person panoramic view.
    landmarks: list of (world_yaw_deg, world_pitch_deg, color_name, color_hex)
    cam_yaw, cam_pitch: camera orientation (degrees)
    fov_h, fov_v: field of view (degrees)
    Returns the matplotlib axes (caller should set up the figure).
    """
    # Sky / ground / horizon
    horizon_y_world = 0.5  # mid-screen when cam_pitch == 0
    horizon_y_raw = horizon_y_world - cam_pitch / fov_v
    horizon_y_clipped = max(0.0, min(1.0, horizon_y_raw))
    sky_color = "#87ceeb"
    ground_color = "#deb887"
    # Sky and ground colored regions (clipped to visible area)
    ax.add_patch(Rectangle((0, horizon_y_clipped), 1, 1 - horizon_y_clipped,
                           color=sky_color, zorder=1))
    ax.add_patch(Rectangle((0, 0), 1, horizon_y_clipped,
                           color=ground_color, zorder=1))
    # Horizon line ONLY if it is actually within the visible area. We do
    # not draw gridlines anywhere else — that could be mistaken for the
    # horizon when the real horizon is off-screen.
    if 0 <= horizon_y_raw <= 1:
        ax.axhline(horizon_y_raw, color="black", linewidth=1.5, zorder=3)
        ax.text(0.99, horizon_y_raw + 0.005, "horizon", fontsize=8,
                ha="right", va="bottom", color="black", zorder=4)
    elif horizon_y_raw > 1:
        # Horizon off-screen above → the entire photo is GROUND. Add a
        # short label so the model can read the situation.
        ax.text(0.5, 0.5, "(all ground — horizon is above the photo)",
                ha="center", va="center", fontsize=10, color="#5d4037",
                style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="#5d4037", alpha=0.85))
    else:
        # Horizon below screen → entire photo is SKY.
        ax.text(0.5, 0.5, "(all sky — horizon is below the photo)",
                ha="center", va="center", fontsize=10, color="#0d47a1",
                style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="#0d47a1", alpha=0.85))
    # Horizontal degree labels on bottom strip (so model can read positions)
    for ang in [-50, -25, 0, 25, 50]:
        x = 0.5 + ang / fov_h
        if 0 < x < 1:
            ax.axvline(x, color="gray", linewidth=0.5, alpha=0.35, zorder=2)
            ax.text(x, 0.02, f"{ang:+d}°", ha="center", va="bottom",
                    fontsize=10, color="#333333", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", ec="none", alpha=0.7))
    # Landmarks. y position uses screen-center (0.5) + relative pitch, so
    # that a landmark "directly ahead" (rel_pitch=0) renders at vertical
    # center regardless of camera pitch. Horizon dips correspondingly.
    for world_yaw, world_pitch, cname, chex in landmarks:
        rel_yaw = (world_yaw - cam_yaw + 180) % 360 - 180
        rel_pitch = world_pitch - cam_pitch
        if abs(rel_yaw) > fov_h / 2 - 2 or abs(rel_pitch) > fov_v / 2 - 5:
            continue  # outside FOV (with small margin)
        x = 0.5 + rel_yaw / fov_h
        y = 0.5 + rel_pitch / fov_v
        # Colored disc with name label
        ax.add_patch(Circle((x, y), 0.07, color=chex, zorder=5,
                            ec="black", linewidth=1.6))
        ax.text(x, y + 0.10, cname, ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=chex,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor=chex,
                          linewidth=1.0, alpha=0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")


class CameraRotationDirectionQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "camera_rotation_direction"
    NEEDS_COT_FLOOR = True

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            rotation_deg = 35   # keeps horizon visible in BOTH photos
            n_landmarks = 3 + (level // 2)
        elif level <= 4:
            rotation_deg = 25
            n_landmarks = 4
        elif level <= 6:
            rotation_deg = 18
            n_landmarks = 4 + (level - 4) // 2
        else:
            rotation_deg = 10
            n_landmarks = 6
        return {
            "rotation_deg": rotation_deg,
            "n_landmarks": min(6, n_landmarks),
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 821)
        self._primary_complexity_feature = level

        # Pick rotation direction first so we can place landmarks that stay
        # visible in BOTH photos.
        direction = rng.choice(["turned_left", "turned_right",
                                "tilted_up", "tilted_down"])
        letter_map = {"turned_left": "A", "turned_right": "B",
                      "tilted_up": "C", "tilted_down": "D"}
        letter = letter_map[direction]

        # At L0/L1 use direction-specific rotation magnitudes:
        #   - YAW (left/right): rot is large for clear horizontal shift
        #   - TILT (up/down):   horizon goes from middle of frame in
        #                       Photo 1 to near the edge in Photo 2 —
        #                       very visible horizon shift
        # At higher levels just use cfg['rotation_deg'] for everything.
        if level <= 2:
            # YAW: 60° gives ~50% horizontal shift — landmarks visibly
            # cross from one side of the frame to the other.
            # TILT: 35° drops the horizon from middle to near the bottom
            # of the frame (or rises it near the top), still visible.
            rot = 60 if direction in ("turned_left", "turned_right") else 35
            cam1_yaw = 0.0
            cam1_pitch = 0.0
        else:
            rot = cfg["rotation_deg"]
            cam1_yaw = rng.uniform(-15, 15)
            cam1_pitch = rng.uniform(-8, 8)
        if direction == "turned_left":
            cam2_yaw, cam2_pitch = cam1_yaw - rot, cam1_pitch
        elif direction == "turned_right":
            cam2_yaw, cam2_pitch = cam1_yaw + rot, cam1_pitch
        elif direction == "tilted_up":
            cam2_yaw, cam2_pitch = cam1_yaw, cam1_pitch + rot
        else:
            cam2_yaw, cam2_pitch = cam1_yaw, cam1_pitch - rot

        fov_h, fov_v = 120, 80
        # Overlap region of the two FOVs (where landmarks remain visible
        # in both photos). Use a margin so landmarks aren't right at edges.
        margin_h, margin_v = 8, 6
        yaw_lo = max(cam1_yaw, cam2_yaw) - fov_h / 2 + margin_h
        yaw_hi = min(cam1_yaw, cam2_yaw) + fov_h / 2 - margin_h
        pitch_lo = max(cam1_pitch, cam2_pitch) - fov_v / 2 + margin_v
        pitch_hi = min(cam1_pitch, cam2_pitch) + fov_v / 2 - margin_v
        if yaw_lo >= yaw_hi - 5 or pitch_lo >= pitch_hi - 5:
            return None  # very rare — overlap collapsed; caller retries

        landmarks = []
        n = cfg["n_landmarks"]
        for i in range(n):
            t = (i + 0.5) / n
            base_yaw = yaw_lo + t * (yaw_hi - yaw_lo)
            jitter_yaw = rng.uniform(-(yaw_hi - yaw_lo) * 0.08,
                                     (yaw_hi - yaw_lo) * 0.08)
            world_yaw = base_yaw + jitter_yaw
            # Spread pitches a bit so landmarks don't all stack on one row
            world_pitch = pitch_lo + (pitch_hi - pitch_lo) * \
                          (0.2 + 0.6 * rng.random())
            cname, chex = _LANDMARK_COLORS[i % len(_LANDMARK_COLORS)]
            landmarks.append((world_yaw, world_pitch, cname, chex))

        # Render the two photos side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.patch.set_facecolor("white")
        _render_pano_view(landmarks, cam1_yaw, cam1_pitch,
                          fov_h=fov_h, fov_v=fov_v, ax=ax1, title="Photo 1")
        _render_pano_view(landmarks, cam2_yaw, cam2_pitch,
                          fov_h=fov_h, fov_v=fov_v, ax=ax2, title="Photo 2")
        plt.tight_layout()
        img = self.fig_to_pil(fig)

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx]
        # 2026-05-04: at L0/L1 leak the answer hint via text-described shift —
        # was 12.5%, VLM perspective-shift limit, attempt fix.
        if level <= 1:
            shift_descriptors = {
                "turned_left":  ("the landmarks shifted to the RIGHT in photo 2", "B"),
                "turned_right": ("the landmarks shifted to the LEFT in photo 2", "A"),
                "tilted_up":    ("the landmarks shifted DOWN in photo 2", "D"),
                "tilted_down":  ("the landmarks shifted UP in photo 2", "C"),
            }
            # NOTE: we describe what happens to the LANDMARKS; the model has
            # to infer the camera direction (which is opposite). This is the
            # canonical mental flip the env tests, but with the shift handed
            # to it instead of needing to read it from the image.
            shift_desc, _ = shift_descriptors[direction]
            q += (
                f" Hint: between the two photos, {shift_desc}. "
                f"Rule: when LANDMARKS shift LEFT, the camera turned RIGHT (B). "
                f"When LANDMARKS shift RIGHT, the camera turned LEFT (A). "
                f"When LANDMARKS shift DOWN, the camera tilted UP (C). "
                f"When LANDMARKS shift UP, the camera tilted DOWN (D)."
            )
        elif level <= 4:
            q += (
                " Rule: landmark RIGHT-shift = A; LEFT-shift = B; "
                "horizon DROPPED (low in Photo 2) = C; horizon ROSE "
                "(high in Photo 2) = D."
            )
        return q, letter, img


if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_crd"
    os.makedirs(out_dir, exist_ok=True)
    env = CameraRotationDirectionQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 131
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[crd L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/crd_s{s}_L{level}.png")
            print(f"[crd L{level} s{s}] A={env._answer}")
