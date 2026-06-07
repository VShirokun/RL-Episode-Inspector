// Helpers to pull reward series out of a loaded episode and assign stable
// colors. Pure functions so they can be tested without rendering.

import type { LoadedEpisode } from "../types/episode";
import type { AgentSpec, SignalKind } from "../types/signal";
import type { Series } from "./TimeSeriesChart";

const PALETTE = [
  "#4f9dff", "#ff6b6b", "#4ecb71", "#ffb454", "#b78bff", "#3fd0c9",
  "#ff8fcf", "#c9d44e", "#7a8cff", "#ff9d5c",
];

export function colorFor(_name: string, index: number): string {
  return PALETTE[index % PALETTE.length];
}

function numericColumn(loaded: LoadedEpisode, name: string): number[] | null {
  const col = loaded.columns[name];
  if (!col) return null;
  return (col as Array<number | boolean>).map((v) => (typeof v === "number" ? v : v ? 1 : 0));
}

/** Agents declared by a (multi-agent) episode; [] for single-agent episodes. */
export function agentsOf(loaded: LoadedEpisode): AgentSpec[] {
  return loaded.metadata.agents ?? [];
}

/**
 * Names of weighted reward component signals for an agent. ``agent === null``
 * selects the single-agent / shared signals (signal.agent == null); a string
 * selects that agent's signals.
 */
export function weightedRewardNames(loaded: LoadedEpisode, agent: string | null = null): string[] {
  return loaded.metadata.signals
    .filter((s) => s.kind === "reward_weighted" && (s.agent ?? null) === agent)
    .map((s) => s.name);
}

export function rawRewardNames(loaded: LoadedEpisode, agent: string | null = null): string[] {
  return loaded.metadata.signals
    .filter((s) => s.kind === "reward_raw" && (s.agent ?? null) === agent)
    .map((s) => s.name);
}

/** Column name of an agent's per-frame total (or the team/single-agent total). */
export function stepTotalName(agent: string | null = null): string {
  return agent ? `reward_${agent}_step_total` : "reward_step_total";
}

/** Column name of an agent's cumulative return (or the team/single-agent one). */
export function cumulativeName(agent: string | null = null): string {
  return agent ? `reward_${agent}_cumulative` : "reward_cumulative";
}

/** Names of all signals of a given kind, in declaration order (e.g. actions). */
export function signalNamesByKind(loaded: LoadedEpisode, kind: SignalKind): string[] {
  return loaded.metadata.signals.filter((s) => s.kind === kind).map((s) => s.name);
}

/** Map signal name -> unit, for chart titles/readouts (only signals with units). */
export function signalUnits(loaded: LoadedEpisode): Map<string, string> {
  const units = new Map<string, string>();
  for (const s of loaded.metadata.signals) if (s.unit) units.set(s.name, s.unit);
  return units;
}

/** Build chart Series for a list of signal names, skipping any that are absent. */
export function buildSeries(loaded: LoadedEpisode, names: string[]): Series[] {
  const out: Series[] = [];
  names.forEach((name, i) => {
    const values = numericColumn(loaded, name);
    if (values) out.push({ name: prettyName(name), color: colorFor(name, i), values });
  });
  return out;
}

/** Strip the reward_ prefix / _weighted suffix for compact legends. */
export function prettyName(name: string): string {
  return name
    .replace(/^reward_/, "")
    .replace(/_weighted$/, "")
    .replace(/_raw$/, " (raw)")
    .replace(/_/g, " ");
}
