// Episode list + best/worst/median quick-select. Shows the summary fields that
// matter when triaging episodes (return, length, why it ended).

import { usePlaybackStore } from "../playback/playbackStore";

function endedTag(terminated: boolean, truncated: boolean): string {
  if (terminated) return "terminated";
  if (truncated) return "truncated";
  return "—";
}

export function EpisodeSelector() {
  const episodes = usePlaybackStore((s) => s.episodes);
  const selectedId = usePlaybackStore((s) => s.selectedId);
  const selectEpisode = usePlaybackStore((s) => s.selectEpisode);
  const selectRanked = usePlaybackStore((s) => s.selectRanked);
  const listError = usePlaybackStore((s) => s.listError);

  return (
    <div className="episode-selector panel">
      <div className="panel-header">
        <h3>Episodes</h3>
        <div className="ranking-buttons">
          <button className="button" onClick={() => selectRanked("best")} data-testid="btn-best">
            Best
          </button>
          <button className="button" onClick={() => selectRanked("median")}>
            Median
          </button>
          <button className="button" onClick={() => selectRanked("worst")}>
            Worst
          </button>
        </div>
      </div>
      {listError && <div className="inline-error">{listError}</div>}
      <div className="episode-list" data-testid="episode-list">
        {episodes.map((e) => (
          <button
            key={e.episode_id}
            className={`episode-row ${e.episode_id === selectedId ? "selected" : ""}`}
            onClick={() => selectEpisode(e.episode_id)}
            data-episode-id={e.episode_id}
          >
            <span className="ep-id">{e.episode_id}</span>
            <span className="ep-return">{e.episode_return.toFixed(1)}</span>
            <span className="ep-meta">
              {e.num_frames}f · {e.duration_seconds.toFixed(1)}s
            </span>
            <span className={`ep-tag ${endedTag(e.terminated, e.truncated)}`}>
              {endedTag(e.terminated, e.truncated)}
              {e.reset_reason ? ` (${e.reset_reason})` : ""}
            </span>
          </button>
        ))}
        {episodes.length === 0 && !listError && (
          <div className="empty">No episodes. Generate some with the CLI, then refresh.</div>
        )}
      </div>
    </div>
  );
}
