import { describe, expect, it } from "vitest";
import {
  agentsOf,
  buildSeries,
  cumulativeName,
  signalNamesByKind,
  signalUnits,
  stepTotalName,
  weightedRewardNames,
} from "./rewardSeries";
import type { LoadedEpisode } from "../types/episode";
import type { AgentSpec, SignalKind, SignalSpec } from "../types/signal";

function sig(
  name: string,
  kind: SignalKind,
  unit: string | null = null,
  agent: string | null = null,
): SignalSpec {
  return { name, kind, dtype: "float32", shape: [], unit, description: null, agent };
}

function episode(
  signals: SignalSpec[],
  columns: Record<string, number[]>,
  agents: AgentSpec[] = [],
): LoadedEpisode {
  return {
    metadata: {
      schema_version: "0.1.0", episode_id: "e", run_id: null, task_name: "T",
      task_source: "test", env_id: 0, episode_index: 0, created_at: "",
      num_frames: 3, dt: 1 / 30, fps: 30, duration_seconds: 0.1,
      global_step_start: 0, global_step_end: 2, terminated: false, truncated: true,
      reset_reason: null, episode_return: 0, agents, policy_checkpoint: null, seed: null,
      signals, viewer: { type: "cartpole", state_mapping: {} },
    },
    columns,
  };
}

const ep = episode(
  [
    sig("cmd_x", "action"),
    sig("ee_speed", "action", "m/s"),
    sig("ee_x", "state", "m"),
    sig("reward_alive_weighted", "reward_weighted"),
  ],
  { cmd_x: [0, 1, 2], ee_speed: [0.1, 0.2, 0.3], ee_x: [1, 1, 1] },
);

describe("signalNamesByKind", () => {
  it("returns names of a kind in declaration order", () => {
    expect(signalNamesByKind(ep, "action")).toEqual(["cmd_x", "ee_speed"]);
    expect(signalNamesByKind(ep, "state")).toEqual(["ee_x"]);
  });

  it("returns [] for a kind with no signals", () => {
    expect(signalNamesByKind(ep, "observation")).toEqual([]);
  });

  it("does not leak reward signals into generic kinds", () => {
    expect(signalNamesByKind(ep, "action")).not.toContain("reward_alive_weighted");
    expect(weightedRewardNames(ep)).toEqual(["reward_alive_weighted"]);
  });
});

describe("signalUnits", () => {
  it("maps only signals that declare a unit", () => {
    const units = signalUnits(ep);
    expect(units.get("ee_speed")).toBe("m/s");
    expect(units.get("ee_x")).toBe("m");
    expect(units.has("cmd_x")).toBe(false);
  });
});

describe("buildSeries", () => {
  it("skips names with no column and prettifies the rest", () => {
    const series = buildSeries(ep, ["cmd_x", "missing", "ee_speed"]);
    expect(series.map((s) => s.name)).toEqual(["cmd x", "ee speed"]);
    expect(series[0].values).toEqual([0, 1, 2]);
  });
});

describe("multi-agent reward helpers", () => {
  const marl = episode(
    [
      sig("reward_cart_pole_pos_weighted", "reward_weighted", null, "cart"),
      sig("reward_pendulum_pos_weighted", "reward_weighted", null, "pendulum"),
      sig("reward_step_total", "reward_total"), // team (agent null)
    ],
    {},
    [{ id: "cart", label: "Cart" }, { id: "pendulum", label: "Pendulum" }],
  );

  it("agentsOf returns declared agents ([] for single-agent)", () => {
    expect(agentsOf(marl).map((a) => a.id)).toEqual(["cart", "pendulum"]);
    expect(agentsOf(ep)).toEqual([]);
  });

  it("weightedRewardNames filters by agent (null = shared/single-agent)", () => {
    expect(weightedRewardNames(marl, "cart")).toEqual(["reward_cart_pole_pos_weighted"]);
    expect(weightedRewardNames(marl, "pendulum")).toEqual(["reward_pendulum_pos_weighted"]);
    expect(weightedRewardNames(marl, null)).toEqual([]); // no shared weighted terms
    // single-agent: signals have agent null, default arg selects them
    expect(weightedRewardNames(ep)).toEqual(["reward_alive_weighted"]);
  });

  it("stepTotalName / cumulativeName namespace per agent", () => {
    expect(stepTotalName("cart")).toBe("reward_cart_step_total");
    expect(cumulativeName("pendulum")).toBe("reward_pendulum_cumulative");
    expect(stepTotalName()).toBe("reward_step_total");
    expect(cumulativeName()).toBe("reward_cumulative");
  });
});
