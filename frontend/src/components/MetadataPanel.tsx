// Compact episode metadata summary (return, length, termination, seed, ...).

import { usePlaybackStore } from "../playback/playbackStore";

export function MetadataPanel() {
  const loaded = usePlaybackStore((s) => s.loaded);
  if (!loaded) return null;
  const m = loaded.metadata;

  const ended = m.terminated ? "terminated" : m.truncated ? "truncated" : "running";
  const rows: [string, string][] = [
    ["episode", m.episode_id],
    ["task", `${m.task_name} (${m.task_source})`],
    ["return", m.episode_return.toFixed(3)],
    ["frames", `${m.num_frames} @ ${m.fps.toFixed(0)} fps`],
    ["duration", `${m.duration_seconds.toFixed(2)} s`],
    ["ended", `${ended}${m.reset_reason ? ` · ${m.reset_reason}` : ""}`],
    ["env / seed", `${m.env_id} / ${m.seed ?? "—"}`],
  ];

  return (
    <div className="metadata-panel">
      {rows.map(([k, v]) => (
        <div className="meta-row" key={k}>
          <span className="k">{k}</span>
          <span className="v">{v}</span>
        </div>
      ))}
    </div>
  );
}
