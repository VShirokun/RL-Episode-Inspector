"""Train on Cartpole while periodically recording episodes (template).

Recording during training is the same recorder hook used elsewhere: inside your
rollout/eval loop, for the environment index you want to inspect, call
``recorder.record_frame(...)`` each control step and ``recorder.end_episode(...)``
when that env resets. To avoid flooding disk, record only every Nth episode or
cap with ``EpisodeRecorder(max_saved_episodes=...)``.

This file is a documented template, not a runnable trainer, because the choice of
RL framework (rsl_rl, skrl, rl_games, sb3) is yours. The verified demo path is
``generate_demo_episodes.py``. See docs/isaac_lab_integration.md.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit(
        "train_cartpole_with_recorder.py is a documented template — see its module "
        "docstring and generate_demo_episodes.py for the working recorder hook."
    )
