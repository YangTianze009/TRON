"""Compatibility shim — historical envs (bar_chart, line_chart,
coordinate_geometry, histogram, radar_chart, scatter_plot, triangle_property)
import StandaloneVisualEnv from `.base`. The canonical implementation lives
in `.standalone_base` and includes the v6 dash-scaling rcParam fix,
retry-on-render-failure logic, and continuous reward shaping. Re-exporting
the same class here keeps those legacy importers in sync with the upgraded
base without per-file changes.

Note: the original base.py defined two extra helpers (_to_float,
_try_eval_math) that no env in the registry currently calls — verified by
grep before the swap. If a future env needs them, copy from git history.
"""
from .standalone_base import StandaloneVisualEnv  # noqa: F401
