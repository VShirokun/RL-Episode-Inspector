// Generic per-kind signal panel: tabs over the episode's non-reward signal
// kinds (Actions / Observations / State / Debug / Events) and small-multiple
// time-series for the selected kind. Reuses the same TimeSeriesChart + series
// helpers as the reward panels, so it's robot-agnostic — it shows whatever the
// recorder saved (the Franka Reach demo records cmd_x/y/z + ee_speed as actions,
// ee/target positions + distance as state). Rewards keep their dedicated panels.

import { useMemo, useState } from "react";
import { usePlaybackStore } from "../playback/playbackStore";
import { TimeSeriesChart } from "./TimeSeriesChart";
import { buildSeries, prettyName, signalNamesByKind, signalUnits } from "./rewardSeries";
import type { SignalKind } from "../types/signal";

// Kinds this panel surfaces, in tab order. Reward kinds + pose are intentionally
// excluded (rewards have their own panels; pose drives the 3D viewer).
const KIND_TABS: { kind: SignalKind; label: string }[] = [
  { kind: "action", label: "Actions" },
  { kind: "observation", label: "Observations" },
  { kind: "state", label: "State" },
  { kind: "debug", label: "Debug" },
  { kind: "event", label: "Events" },
];

const COLORS = ["#4f9dff", "#4ecb71", "#ffb454", "#b78bff", "#ff6b6b", "#3fd0c9"];

export function SignalCharts() {
  const loaded = usePlaybackStore((s) => s.loaded);
  const currentFrame = usePlaybackStore((s) => s.currentFrame);
  const numFrames = usePlaybackStore((s) => s.numFrames());
  const seek = usePlaybackStore((s) => s.seek);
  const [active, setActive] = useState<SignalKind | null>(null);

  // Only show tabs for kinds this episode actually has.
  const tabs = useMemo(
    () => (loaded ? KIND_TABS.filter((t) => signalNamesByKind(loaded, t.kind).length > 0) : []),
    [loaded],
  );
  // Derive the selected tab so it stays valid when the episode (and its kinds)
  // changes, without a sync effect: fall back to the first available tab.
  const selected = active && tabs.some((t) => t.kind === active) ? active : tabs[0]?.kind ?? null;

  const charts = useMemo(() => {
    if (!loaded || !selected) return [];
    const units = signalUnits(loaded);
    return signalNamesByKind(loaded, selected)
      .map((name, i) => {
        const series = buildSeries(loaded, [name]);
        if (series.length) series[0].color = COLORS[i % COLORS.length];
        const unit = units.get(name);
        return { name, label: prettyName(name) + (unit ? ` (${unit})` : ""), series };
      })
      .filter((c) => c.series.length > 0);
  }, [loaded, selected]);

  if (!loaded || tabs.length === 0) return null;

  return (
    <div className="signal-charts panel">
      <div className="panel-header">
        <h3>Signals</h3>
        <div className="seg-group" role="tablist" aria-label="Signal kind">
          {tabs.map((t) => (
            <button
              key={t.kind}
              role="tab"
              aria-selected={selected === t.kind}
              className={`button seg ${selected === t.kind ? "active" : ""}`}
              onClick={() => setActive(t.kind)}
              data-testid={`signal-tab-${t.kind}`}
            >
              {t.label}
            </button>
          ))}
        </div>
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
