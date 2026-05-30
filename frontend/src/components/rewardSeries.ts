// Helpers to pull reward series out of a loaded episode and assign stable
// colors. Pure functions so they can be tested without rendering.

import type { LoadedEpisode } from "../types/episode";
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

/** Names of weighted reward component signals, e.g. reward_alive_weighted. */
export function weightedRewardNames(loaded: LoadedEpisode): string[] {
  return loaded.metadata.signals
    .filter((s) => s.kind === "reward_weighted")
    .map((s) => s.name);
}

export function rawRewardNames(loaded: LoadedEpisode): string[] {
  return loaded.metadata.signals.filter((s) => s.kind === "reward_raw").map((s) => s.name);
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
