// Per-component reward charts (small multiples): total, cumulative, and each
// weighted reward term on its own axis. Raw terms are available but hidden by
// default to avoid clutter (toggle at the top).

import { useMemo, useState } from "react";
import { usePlaybackStore } from "../playback/playbackStore";
import { TimeSeriesChart, type Series } from "./TimeSeriesChart";
import {
  buildSeries,
  cumulativeName,
  prettyName,
  rawRewardNames,
  stepTotalName,
  weightedRewardNames,
} from "./rewardSeries";

const SMALL_COLORS = ["#4f9dff", "#4ecb71", "#ffb454", "#b78bff", "#ff6b6b", "#3fd0c9"];

/** ``agent`` (multi-agent) restricts charts to that agent's reward components. */
export function RewardCharts({ agent = null }: { agent?: string | null }) {
  const loaded = usePlaybackStore((s) => s.loaded);
  const currentFrame = usePlaybackStore((s) => s.currentFrame);
  const numFrames = usePlaybackStore((s) => s.numFrames());
  const seek = usePlaybackStore((s) => s.seek);
  const [showRaw, setShowRaw] = useState(false);

  // Build the per-chart series once per episode (and raw-toggle), not every
  // frame, so TimeSeriesChart can keep its memoized paths during playback.
  const charts = useMemo(() => {
    if (!loaded) return [];
    const weighted = weightedRewardNames(loaded, agent);
    const total = stepTotalName(agent);
    const cumulative = cumulativeName(agent);
    const names = [total, cumulative, ...weighted, ...(showRaw ? rawRewardNames(loaded, agent) : [])];
    const out: { name: string; label: string; series: Series[] }[] = [];
    names.forEach((name, i) => {
      const series = buildSeries(loaded, [name]);
      if (series.length === 0) return;
      series[0].color = SMALL_COLORS[i % SMALL_COLORS.length];
      const label =
        name === total
          ? "step total"
          : name === cumulative
            ? "cumulative"
            : prettyName(name) + (weighted.includes(name) ? " (weighted)" : "");
      out.push({ name, label, series });
    });
    return out;
  }, [loaded, showRaw, agent]);

  if (!loaded) return null;

  return (
    <div className="reward-charts panel">
      <div className="panel-header">
        <h3>Reward components</h3>
        <label className="toggle">
          <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} />
          show raw terms
        </label>
      </div>
      <div className="reward-charts-scroll">
        {charts.map((c) => (
          <TimeSeriesChart
            key={c.name}
            title={c.label}
            series={c.series}
            numFrames={numFrames}
            currentFrame={currentFrame}
            onSeek={seek}
            height={92}
            showLegend={false}
          />
        ))}
      </div>
    </div>
  );
}
