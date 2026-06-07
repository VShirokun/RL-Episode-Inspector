"""Command-line interface: ``rl-episode-inspector <command>``."""

from __future__ import annotations

from pathlib import Path

import typer

from . import __version__
from .fake_data import generate_fake_cartpole_episodes
from .ranking import EpisodeRanker

app = typer.Typer(
    add_completion=False,
    help="Record, store, rank and replay RL episodes from Isaac Lab.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("generate-fake-cartpole")
def generate_fake_cartpole(
    output_dir: Path = typer.Option(
        Path("sample_data/cartpole/episodes"), "--output-dir", "-o"
    ),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n", min=1),
    seed: int = typer.Option(42, "--seed"),
    max_frames: int = typer.Option(400, "--max-frames", min=1),
    fps: int = typer.Option(60, "--fps", min=1),
) -> None:
    """Generate fake Cartpole-like episodes (no Isaac Lab required)."""
    created = generate_fake_cartpole_episodes(
        output_dir,
        num_episodes=num_episodes,
        seed=seed,
        max_frames=max_frames,
        fps=fps,
    )
    typer.echo(f"Generated {len(created)} episodes in {output_dir}")
    for episode_id in created:
        typer.echo(f"  {episode_id}")


@app.command()
def rank(
    episodes_dir: Path = typer.Argument(Path("sample_data/cartpole/episodes")),
) -> None:
    """Print best / worst / median episodes for an episodes directory."""
    ranker = EpisodeRanker(episodes_dir)
    episodes = ranker.list_episodes()
    if not episodes:
        typer.echo("No episodes found.")
        raise typer.Exit(code=1)
    typer.echo(f"{len(episodes)} episodes:")
    for s in episodes:
        flag = "T" if s.terminated else ("Tr" if s.truncated else "-")
        typer.echo(f"  {s.episode_id:>20}  return={s.episode_return:10.3f}  [{flag}]")
    for label, summary in (
        ("best", ranker.get_best()),
        ("median", ranker.get_median()),
        ("worst", ranker.get_worst()),
    ):
        if summary:
            typer.echo(f"{label:>7}: {summary.episode_id} (return={summary.episode_return:.3f})")


@app.command()
def serve(
    episodes_dir: Path = typer.Option(
        # Default to the real Isaac Lab Franka Reach experiment (committed, so it
        # serves out of the box without Isaac). Override with --episodes-dir.
        Path("sample_data/reach/episodes"), "--episodes-dir", "-d"
    ),
    assets_dir: Path | None = typer.Option(
        None, "--assets-dir", help="robot mesh GLBs served at /assets"
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the local backend API server."""
    import uvicorn

    from .server import create_app

    server_app = create_app(episodes_dir, assets_dir=assets_dir)
    typer.echo(f"Serving episodes from {episodes_dir} at http://{host}:{port}")
    uvicorn.run(server_app, host=host, port=port)


if __name__ == "__main__":
    app()
