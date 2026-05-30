// Transport controls: play/pause, frame stepping, scrub bar, speed, frame info.

import { usePlaybackStore } from "../playback/playbackStore";
import { PLAYBACK_SPEEDS, type PlaybackSpeed } from "../types/episode";

export function TimelineControls() {
  const isPlaying = usePlaybackStore((s) => s.isPlaying);
  const currentFrame = usePlaybackStore((s) => s.currentFrame);
  const numFrames = usePlaybackStore((s) => s.numFrames());
  const speed = usePlaybackStore((s) => s.speed);
  const loaded = usePlaybackStore((s) => s.loaded);
  const { togglePlay, stepFrames, first, last, setFrame, setSpeed } = usePlaybackStore.getState();

  const frame = Math.round(currentFrame);
  const t = loaded ? (frame * loaded.metadata.dt).toFixed(2) : "0.00";
  const disabled = numFrames === 0;

  return (
    <div className="timeline-controls" data-testid="timeline-controls">
      <div className="transport">
        <button className="button" onClick={first} disabled={disabled} title="First frame (Home)">
          ⏮
        </button>
        <button
          className="button"
          onClick={() => stepFrames(-1)}
          disabled={disabled}
          title="Previous frame (←)"
        >
          ◀
        </button>
        <button
          className="button play"
          onClick={togglePlay}
          disabled={disabled}
          data-testid="play-pause"
          title="Play/Pause (Space)"
        >
          {isPlaying ? "❚❚" : "►"}
        </button>
        <button
          className="button"
          onClick={() => stepFrames(1)}
          disabled={disabled}
          title="Next frame (→)"
        >
          ▶
        </button>
        <button className="button" onClick={last} disabled={disabled} title="Last frame (End)">
          ⏭
        </button>
      </div>

      <input
        className="scrubber"
        type="range"
        min={0}
        max={Math.max(0, numFrames - 1)}
        value={frame}
        onChange={(e) => setFrame(Number(e.target.value))}
        disabled={disabled}
        data-testid="scrubber"
        aria-label="Timeline scrubber"
      />

      <div className="frame-info">
        <span data-testid="frame-counter">
          {frame} / {Math.max(0, numFrames - 1)}
        </span>
        <span className="muted">{t}s</span>
      </div>

      <label className="speed">
        speed
        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value) as PlaybackSpeed)}
          data-testid="speed-select"
        >
          {PLAYBACK_SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
