// Thin typed wrapper over the backend REST API. Uses same-origin paths so the
// Vite dev proxy (or a reverse proxy in prod) forwards to the Python backend.

import type {
  EpisodeMetadata,
  EpisodeSummary,
  FramesResponse,
} from "../types/episode";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function getJson<T>(url: string): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(url);
  } catch (e) {
    throw new ApiError(`Network error contacting backend: ${String(e)}`, 0);
  }
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new ApiError(detail, resp.status);
  }
  return (await resp.json()) as T;
}

export const api = {
  listEpisodes: () =>
    getJson<{ episodes: EpisodeSummary[] }>("/api/episodes").then((r) => r.episodes),

  getMetadata: (episodeId: string) =>
    getJson<EpisodeMetadata>(`/api/episodes/${encodeURIComponent(episodeId)}/metadata`),

  getFrames: (episodeId: string, start?: number, end?: number) => {
    const params = new URLSearchParams();
    if (start != null) params.set("start", String(start));
    if (end != null) params.set("end", String(end));
    const qs = params.toString();
    const suffix = qs ? `?${qs}` : "";
    return getJson<FramesResponse>(
      `/api/episodes/${encodeURIComponent(episodeId)}/frames${suffix}`,
    );
  },

  getRanked: (mode: "best" | "worst" | "median") =>
    getJson<EpisodeSummary>(`/api/ranking?mode=${mode}`),
};
