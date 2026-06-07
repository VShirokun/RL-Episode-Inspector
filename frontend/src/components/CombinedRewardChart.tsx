// All weighted reward components overlaid on a single shared axis, plus the
// step total. Series can be toggled in the legend. This is where you spot which
// components dominate or conflict at a given moment.

import { useMemo, useState } from "react";
import { usePlaybackStore } from "../playback/playbackStore";
import { TimeSeriesChart } from "./TimeSeriesChart";
import { buildSeries, stepTotalName, weightedRewardNames } from "./rewardSeries";

/** ``agent`` (multi-agent) selects that agent's weighted terms + total. */
export function CombinedRewardChart({ agent = null }: { agent?: string | null }) {
  const loaded = usePlaybackStore((s) => s.loaded);
  const currentFrame = usePlaybackStore((s) => s.currentFrame);
  const numFrames = usePlaybackStore((s) => s.numFrames());
  const seek = usePlaybackStore((s) => s.seek);
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const names = useMemo(() => {
    if (!loaded) return [];
    return [stepTotalName(agent), ...weightedRewardNames(loaded, agent)];
  }, [loaded, agent]);

  // Memoize series so their array refs are stable across per-frame re-renders
  // (lets TimeSeriesChart keep its memoized paths instead of rebuilding them).
  const allSeries = useMemo(() => (loaded ? buildSeries(loaded, names) : []), [loaded, names]);
  const visible = useMemo(
    () => allSeries.filter((s) => !hidden.has(s.name)),
    [allSeries, hidden],
  );

  if (!loaded) return null;

  const toggle = (name: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });

  return (
    <div className="combined-chart panel">
      <div className="panel-header">
        <h3>Combined rewards (weighted)</h3>
      </div>
      <TimeSeriesChart
        series={visible}
        numFrames={numFrames}
        currentFrame={currentFrame}
        onSeek={seek}
        height={200}
        showLegend={false}
      />
      <div className="chart-legend interactive">
        {allSeries.map((s) => (
          <button
            key={s.name}
            className={`legend-item button ${hidden.has(s.name) ? "off" : ""}`}
            onClick={() => toggle(s.name)}
            title="Toggle series"
          >
            <span className="swatch" style={{ background: s.color }} />
            {s.name}
          </button>
        ))}
      </div>
    </div>
  );
}
