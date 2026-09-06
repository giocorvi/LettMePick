export type ModelSummary = {
  id: string;
  name: string;
  description: string | null;
  has_config: boolean;
  active: boolean;
  loading: boolean;
  loading_stage: string | null;
};

export type Movie = {
  id: string;
  title: string;
  year: number;
  genres: string[];
};

export type ContextMovie = Movie & { rating: number };
export type RankedMovie = Movie & { score: number; score_five: number };

export type SavedContext = {
  id: string;
  name: string;
  model_id: string;
  model_name: string;
  movies: ContextMovie[];
  updated_at: string;
};

export type AutoResponse = {
  run_id: string;
  batches: number;
  sampled: number;
  remaining: number;
  complete: boolean;
  results: RankedMovie[];
};

export type MovieFacets = {
  min_year: number;
  max_year: number;
  genres: string[];
  popularity_levels: PopularityLevel[];
};

export type PopularityLevel = {
  id: string;
  label: string;
  min_rating_count: number;
};

export type LoadedModel = {
  id: string;
  name: string;
  description: string | null;
  catalog_size: number;
  device: string;
  architecture: Record<string, unknown>;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  models: () => request<{ models: ModelSummary[] }>("/api/models"),
  loadModel: (id: string) => request<LoadedModel>(`/api/models/${id}/load`, { method: "POST" }),
  inspectConfig: (id: string) => request<{ content: string }>(`/api/models/${id}/config`),
  contexts: () => request<{ contexts: SavedContext[] }>("/api/contexts"),
  saveContext: (name: string, modelId: string, context: ContextMovie[]) =>
    request<SavedContext>("/api/contexts", {
      method: "POST",
      body: JSON.stringify({
        name,
        model_id: modelId,
        context: context.map(({ id, rating }) => ({ movie_id: id, rating })),
      }),
    }),
  deleteContext: (id: string) =>
    request<void>(`/api/contexts/${id}`, { method: "DELETE" }),
  searchMovies: (query: string) =>
    request<{ movies: Movie[] }>(`/api/movies?query=${encodeURIComponent(query)}&limit=20`),
  movieFacets: () => request<MovieFacets>("/api/movie-facets"),
  manual: (context: ContextMovie[], query: Movie[]) =>
    request<{ results: RankedMovie[] }>("/api/predictions/manual", {
      method: "POST",
      body: JSON.stringify({
        context: context.map(({ id, rating }) => ({ movie_id: id, rating })),
        query_movie_ids: query.map(({ id }) => id),
      }),
    }),
  startAuto: (
    context: ContextMovie[],
    batchSize: number,
    filters: {
      minYear: number | null;
      maxYear: number | null;
      genres: string[];
      minRatingCount: number;
    },
  ) =>
    request<AutoResponse>("/api/auto-runs", {
      method: "POST",
      body: JSON.stringify({
        context: context.map(({ id, rating }) => ({ movie_id: id, rating })),
        batch_size: batchSize,
        min_year: filters.minYear,
        max_year: filters.maxYear,
        genres: filters.genres,
        min_rating_count: filters.minRatingCount,
      }),
    }),
  nextAuto: (runId: string) =>
    request<AutoResponse>(`/api/auto-runs/${runId}/next`, { method: "POST" }),
  stopAuto: (runId: string) =>
    request<void>(`/api/auto-runs/${runId}`, { method: "DELETE" }),
};
