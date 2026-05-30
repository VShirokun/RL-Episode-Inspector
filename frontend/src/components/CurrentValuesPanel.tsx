// Live readout of the current frame: state signals and per-frame reward values.

import { columnValue, usePlaybackStore } from "../playback/playbackStore";

export function CurrentValuesPanel() {
  const loaded = usePlaybackStore((s) => s.loaded);
  const currentFrame = usePlaybackStore((s) => s.currentFrame);
  if (!loaded) return null;
  const frame = Math.round(currentFrame);

  const states = loaded.metadata.signals.filter((s) => s.kind === "state" || s.kind === "action");
  const total = columnValue(loaded, "reward_step_total", frame);
  const cumulative = columnValue(loaded, "reward_cumulative", frame);

  return (
    <div className="values-panel panel">
      <div className="panel-header">
        <h3>Current frame</h3>
      </div>
      <div className="values-grid">
        {states.map((s) => (
          <div className="value-row" key={s.name}>
            <span className="k">{s.name}</span>
            <span className="v">
              {columnValue(loaded, s.name, frame)?.toFixed(3)}
              {s.unit ? <span className="unit"> {s.unit}</span> : null}
            </span>
          </div>
        ))}
        <div className="value-row highlight">
          <span className="k">reward (step)</span>
          <span className="v">{total?.toFixed(3)}</span>
        </div>
        <div className="value-row highlight">
          <span className="k">reward (cumulative)</span>
          <span className="v">{cumulative?.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}
