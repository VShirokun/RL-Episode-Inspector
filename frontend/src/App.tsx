import { useEffect, useState } from "react";
import { CombinedRewardChart } from "./components/CombinedRewardChart";
import { CurrentValuesPanel } from "./components/CurrentValuesPanel";
import { EpisodeSelector } from "./components/EpisodeSelector";
import { ErrorState } from "./components/ErrorState";
import { HelpModal } from "./components/HelpModal";
import { LoadingState } from "./components/LoadingState";
import { MetadataPanel } from "./components/MetadataPanel";
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
  const [helpOpen, setHelpOpen] = useState(false);

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
            <Viewer3D />
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
