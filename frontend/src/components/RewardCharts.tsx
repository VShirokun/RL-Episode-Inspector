// Per-component reward charts (small multiples): total, cumulative, and each
// weighted reward term on its own axis. Raw terms are available but hidden by
// default to avoid clutter (toggle at the top).

import { useState } from "react";
import { usePlaybackStore } from "../playback/playbackStore";
import { TimeSeriesChart } from "./TimeSeriesChart";
import { buildSeries, prettyName, rawRewardNames, weightedRewardNames } from "./rewardSeries";

export function RewardCharts() {
  const loaded = usePlaybackStore((s) => s.loaded);
  const currentFrame = usePlaybackStore((s) => s.currentFrame);
  const numFrames = usePlaybackStore((s) => s.numFrames());
  const seek = usePlaybackStore((s) => s.seek);
  const [showRaw, setShowRaw] = useState(false);

  if (!loaded) return null;

  const totals = ["reward_step_total", "reward_cumulative"];
  const weighted = weightedRewardNames(loaded);
  const raw = showRaw ? rawRewardNames(loaded) : [];
  const names = [...totals, ...weighted, ...raw];

  return (
    <div className="reward-charts panel">
      <div className="panel-header">
        <h3>Reward components</h3>
        <label className="toggle">
          <input
            type="checkbox"
            checked={showRaw}
            onChange={(e) => setShowRaw(e.target.checked)}
          />
          show raw terms
        </label>
      </div>
      <div className="reward-charts-scroll">
        {names.map((name, i) => {
          const series = buildSeries(loaded, [name]);
          if (series.length === 0) return null;
          const isWeighted = weighted.includes(name);
          const label =
            name === "reward_step_total"
              ? "step total"
              : name === "reward_cumulative"
                ? "cumulative"
                : prettyName(name) + (isWeighted ? " (weighted)" : "");
          // recolor so each small-multiple is visually distinct
          series[0].color = ["#4f9dff", "#4ecb71", "#ffb454", "#b78bff", "#ff6b6b", "#3fd0c9"][
            i % 6
          ];
          return (
            <TimeSeriesChart
              key={name}
              title={label}
              series={series}
              numFrames={numFrames}
              currentFrame={currentFrame}
              onSeek={seek}
              height={92}
              showLegend={false}
            />
          );
        })}
      </div>
    </div>
  );
}
