import { useEffect, useState } from "react";
import { ArticulationViewer } from "./components/ArticulationViewer";
import { CombinedRewardChart } from "./components/CombinedRewardChart";
import { CurrentValuesPanel } from "./components/CurrentValuesPanel";
import { EpisodeSelector } from "./components/EpisodeSelector";
import { ErrorState } from "./components/ErrorState";
import { HelpModal } from "./components/HelpModal";
import { LoadingState } from "./components/LoadingState";
import { MetadataPanel } from "./components/MetadataPanel";
import { ReachViewer } from "./components/ReachViewer";
import { RewardCharts } from "./components/RewardCharts";
import { TimelineControls } from "./components/TimelineControls";
import { Viewer3D } from "./components/Viewer3D";
import { useKeyboardControls } from "./playback/useKeyboardControls";
import { usePlaybackLoop } from "./playback/usePlaybackLoop";
import { usePlaybackStore } from "./playback/playbackStore";

export default function App() {
  usePlaybackLoop();
  useKeyboardControls();

  const fetchEpisodes = usePlaybackStore((s) => s.fetchEpisodes);
  const listLoading = usePlaybackStore((s) => s.listLoading);
  const episodeLoading = usePlaybackStore((s) => s.episodeLoading);
  const episodeError = usePlaybackStore((s) => s.episodeError);
  const loaded = usePlaybackStore((s) => s.loaded);
  const selectedId = usePlaybackStore((s) => s.selectedId);
  const renderMode = usePlaybackStore((s) => s.renderMode);
  const setRenderMode = usePlaybackStore((s) => s.setRenderMode);
  const defaultLights = usePlaybackStore((s) => s.defaultLights);
  const setDefaultLights = usePlaybackStore((s) => s.setDefaultLights);
  const [helpOpen, setHelpOpen] = useState(false);
  const hasBodies = (loaded?.metadata.viewer.bodies?.length ?? 0) > 0;

  useEffect(() => {
    fetchEpisodes();
  }, [fetchEpisodes]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="logo">◉</span> RL Episode Inspector
        </div>
        <MetadataPanel />
        {hasBodies && (
          <div className="render-mode" role="group" aria-label="3D render mode" title="Real meshes vs lightweight cubes">
            <button
              className={`button seg ${renderMode === "models" ? "active" : ""}`}
              onClick={() => setRenderMode("models")}
              data-testid="render-models"
            >
              Models
            </button>
            <button
              className={`button seg ${renderMode === "cubes" ? "active" : ""}`}
              onClick={() => setRenderMode("cubes")}
              data-testid="render-cubes"
            >
              Cubes
            </button>
          </div>
        )}
        {hasBodies && (
          <label
            className="light-toggle"
            title="Add a neutral default light rig. Turn off to see only the lights captured from the source task (if any)."
          >
            <input
              type="checkbox"
              checked={defaultLights}
              onChange={(e) => setDefaultLights(e.target.checked)}
              data-testid="default-lights"
            />
            Default light
          </label>
        )}
        <button className="button ghost" onClick={() => setHelpOpen(true)}>
          ? Help
        </button>
      </header>

      <aside className="sidebar">
        <EpisodeSelector />
        <CurrentValuesPanel />
      </aside>

      <main className="main">
        <section className="viewer-pane">
          {listLoading && !loaded ? (
            <LoadingState label="Loading episodes…" />
          ) : episodeLoading ? (
            <LoadingState label="Loading episode…" />
          ) : episodeError ? (
            <ErrorState
              message={episodeError}
              onRetry={selectedId ? () => usePlaybackStore.getState().selectEpisode(selectedId) : undefined}
            />
          ) : loaded ? (
            loaded.metadata.viewer.type === "articulation3d" ? (
              <ArticulationViewer />
            ) : loaded.metadata.viewer.type === "reach3d" ? (
              <ReachViewer />
            ) : (
              <Viewer3D />
            )
          ) : (
            <div className="state empty">Select an episode to begin.</div>
          )}
        </section>

        <section className="charts-pane">
          {loaded && (
            <>
              <CombinedRewardChart />
              <RewardCharts />
            </>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <TimelineControls />
      </footer>

      <HelpModal open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  );
}
