"""Play a policy on Cartpole while recording episodes.

This is a thin example of the recorder hook for a *policy* rollout. It shares the
exact mechanics of ``generate_demo_episodes.py`` (launch app → make env → per
control step: read state, act, step, compute reward terms, record_frame; on done
end_episode), the only difference being where the action comes from.

To record from a trained checkpoint, replace ``choose_action`` with your policy:

    from rsl_rl.runners import OnPolicyRunner  # or your framework
    policy = load_checkpoint(args.checkpoint)
    action = policy(obs)

For the MVP demo the supported, verified entry point is
``generate_demo_episodes.py`` (PD controller). See docs/isaac_lab_integration.md.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit(
        "play_cartpole_with_recorder.py is a documented template. For the demo run "
        "generate_demo_episodes.py instead (it records real Cartpole episodes with a "
        "PD controller). To record a trained policy, copy the rollout loop from "
        "generate_demo_episodes.py and swap in your policy for the action."
    )
