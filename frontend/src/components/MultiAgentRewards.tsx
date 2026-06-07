// Multi-agent reward view. A tab row (Team + one per agent) over the per-agent
// reward decomposition the recorder saved. "Team" overlays each agent's
// cumulative return on one axis (who's winning / where strategies diverge);
// picking an agent shows that agent's combined + per-component reward charts
// (the same panels as single-agent, scoped via the `agent` prop). Only rendered
// when the episode declares agents — single-agent episodes are untouched.

import { useState } from "react";
import { usePlaybackStore } from "../playback/playbackStore";
import { CombinedRewardChart } from "./CombinedRewardChart";
import { RewardCharts } from "./RewardCharts";
import { TimeSeriesChart, type Series } from "./TimeSeriesChart";
import { agentsOf, buildSeries, colorFor, cumulativeName } from "./rewardSeries";

const TEAM = "__team__";

export function MultiAgentRewards() {
  const loaded = usePlaybackStore((s) => s.loaded);
  const currentFrame = usePlaybackStore((s) => s.currentFrame);
  const numFrames = usePlaybackStore((s) => s.numFrames());
  const seek = usePlaybackStore((s) => s.seek);
  const [tab, setTab] = useState<string>(TEAM);

  if (!loaded) return null;
  const agents = agentsOf(loaded);
  if (agents.length === 0) return null;

  const returns = loaded.metadata.agent_returns ?? {};
  const selected = tab !== TEAM && agents.some((a) => a.id === tab) ? tab : TEAM;
  const labelOf = (id: string) => agents.find((a) => a.id === id)?.label ?? id;

  // Team overlay: one cumulative-return line per agent, distinct colors.
  const teamSeries = agents
    .map((a, i) => {
      const s = buildSeries(loaded, [cumulativeName(a.id)])[0];
      return s ? { ...s, name: labelOf(a.id), color: colorFor(a.id, i) } : null;
    })
    .filter((s): s is Series => s !== null);

  return (
    <div className="multiagent-rewards">
      <div className="agent-selector panel">
        <div className="panel-header">
          <h3>Agents</h3>
          <div className="seg-group" role="tablist" aria-label="Agent">
            <button
              role="tab"
              aria-selected={selected === TEAM}
              className={`button seg ${selected === TEAM ? "active" : ""}`}
              onClick={() => setTab(TEAM)}
              data-testid="agent-tab-team"
            >
              Team
            </button>
            {agents.map((a) => (
              <button
                key={a.id}
                role="tab"
                aria-selected={selected === a.id}
                className={`button seg ${selected === a.id ? "active" : ""}`}
                onClick={() => setTab(a.id)}
                data-testid={`agent-tab-${a.id}`}
              >
                {labelOf(a.id)}
              </button>
            ))}
          </div>
        </div>
        <div className="agent-returns">
          {agents.map((a) => {
            const r = returns[a.id] ?? 0;
            return (
              <span className="agent-return" key={a.id}>
                <span className="k">{labelOf(a.id)}</span>
                <span className={`v ${r >= 0 ? "pos" : "neg"}`}>{r.toFixed(1)}</span>
              </span>
            );
          })}
        </div>
      </div>

      {selected === TEAM ? (
        <div className="combined-chart panel">
          <div className="panel-header">
            <h3>Cumulative return per agent</h3>
          </div>
          <TimeSeriesChart
            series={teamSeries}
            numFrames={numFrames}
            currentFrame={currentFrame}
            onSeek={seek}
            height={200}
            showLegend
          />
        </div>
      ) : (
        <>
          <CombinedRewardChart agent={selected} />
          <RewardCharts agent={selected} />
        </>
      )}
    </div>
  );
}
