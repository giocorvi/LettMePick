import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const model = {
  id: "first-cut",
  name: "First Cut",
  description: "Validation selection",
  has_config: true,
  active: false,
};

const alien = { id: "m1", title: "Alien", year: 1979, genres: ["horror"] };
const arrival = { id: "m4", title: "Arrival", year: 2016, genres: ["science fiction"] };
const originalFetch = globalThis.fetch;
let savedContexts: Array<Record<string, unknown>> = [];

function json(value: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: async () => value,
  } as Response);
}

function installFetch() {
  globalThis.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/models" && !init?.method) return json({ models: [model] });
    if (url === "/api/models/first-cut/load") {
      return json({ id: "first-cut", name: "First Cut", catalog_size: 30, device: "cpu", architecture: {} });
    }
    if (url === "/api/models" && !init?.body) return json({ models: [{ ...model, active: true }] });
    if (url === "/api/models/first-cut/config") return json({ content: "[model]\nfeature_size = 999" });
    if (url === "/api/movie-facets") {
      return json({
        min_year: 1900,
        max_year: 2025,
        genres: ["comedy", "drama", "science fiction"],
        popularity_levels: [
          { id: "any", label: "Any", min_rating_count: 0 },
          { id: "established", label: "Established", min_rating_count: 1_000 },
          { id: "popular", label: "Popular", min_rating_count: 100_000 },
          { id: "blockbuster", label: "Blockbuster", min_rating_count: 1_000_000 },
        ],
      });
    }
    if (url === "/api/contexts" && (!init?.method || init.method === "GET")) {
      return json({ contexts: savedContexts });
    }
    if (url === "/api/contexts" && init?.method === "POST") {
      const payload = JSON.parse(String(init.body));
      const saved = {
        id: "saved-1",
        name: payload.name,
        model_id: payload.model_id,
        model_name: "First Cut",
        movies: payload.context.map(({ movie_id, rating }: { movie_id: string; rating: number }) => ({
          ...(movie_id === alien.id ? alien : arrival),
          rating,
        })),
        updated_at: "2026-08-22T20:00:00Z",
      };
      savedContexts = [saved];
      return json(saved);
    }
    if (url === "/api/contexts/saved-1" && init?.method === "DELETE") {
      savedContexts = [];
      return json(undefined, 204);
    }
    if (url.includes("/api/movies?query=Alien")) return json({ movies: [alien] });
    if (url.includes("/api/movies?query=Arrival")) return json({ movies: [arrival] });
    if (url === "/api/predictions/manual") {
      const results = Array.from({ length: 25 }, (_, index) => ({
        ...arrival,
        id: `result-${index}`,
        title: `Result ${index + 1}`,
        score: 1 - index / 100,
        score_five: 5 - index / 20,
      }));
      return json({ results });
    }
    if (url === "/api/auto-runs") {
      return json({ run_id: "run-1", batches: 1, sampled: 1, remaining: 29, complete: false, results: [{ ...arrival, score: .9, score_five: 4.5 }] });
    }
    if (url === "/api/auto-runs/run-1" && init?.method === "DELETE") {
      return json(undefined, 204);
    }
    if (url === "/api/auto-runs/run-1/next") {
      return new Promise(() => undefined);
    }
    throw new Error(`Unhandled request: ${init?.method || "GET"} ${url}`);
  }) as typeof fetch;
}

async function loadModel() {
  await screen.findByRole("option", { name: "First Cut" });
  fireEvent.click(screen.getByRole("button", { name: /(?:Load|Reload) model/ }));
  await screen.findByText("cpu online");
}

async function addContext() {
  const search = screen.getByRole("combobox", { name: "Find a rated movie" });
  fireEvent.change(search, { target: { value: "Alien" } });
  fireEvent.click(await screen.findByRole("option", { name: /Alien/ }));
}

describe("cinematic model workbench", () => {
  beforeEach(() => {
    savedContexts = [];
    installFetch();
  });
  afterEach(() => {
    cleanup();
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("loads a model and presents its registry override config", async () => {
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Inspect config" }));
    expect(await screen.findByText(/overrides values embedded/)).toBeInTheDocument();
    expect(screen.getByText(/feature_size = 999/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close config" }));
    await loadModel();

    expect(screen.getByText("cpu online")).toBeInTheDocument();
  });

  it("builds a half-star context, scores manual queries, and loads more", async () => {
    render(<App />);
    await loadModel();
    expect(screen.getByText("0/512")).toBeInTheDocument();
    await addContext();

    const rating = screen.getByRole("slider", { name: "Rating for Alien" });
    fireEvent.change(rating, { target: { value: "4.5" } });
    expect(rating).toHaveAttribute("step", "0.5");

    const querySearch = screen.getByRole("combobox", { name: "Add movies to the query slate" });
    fireEvent.change(querySearch, { target: { value: "Arrival" } });
    fireEvent.click(await screen.findByRole("option", { name: /Arrival/ }));
    fireEvent.click(screen.getByRole("button", { name: /Score the slate/ }));

    expect(await screen.findByText("Result 1")).toBeInTheDocument();
    expect(screen.queryByText("Result 21")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Load next 20/ }));
    expect(screen.getByText("Result 21")).toBeInTheDocument();
  });

  it("stores a named context and restores it after remounting", async () => {
    render(<App />);
    await loadModel();
    await addContext();
    fireEvent.change(screen.getByRole("slider", { name: "Rating for Alien" }), { target: { value: "4.5" } });
    fireEvent.click(screen.getByText("Context archive"));
    fireEvent.change(screen.getByRole("textbox", { name: "Save current context" }), { target: { value: "Sunday matinee" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Saved “Sunday matinee” on this machine.")).toBeInTheDocument();
    cleanup();

    render(<App />);
    await loadModel();
    fireEvent.click(screen.getByText("Context archive"));
    const savedOption = screen.getByRole("option", { name: /Sunday matinee · 1 films/ });
    fireEvent.change(screen.getByRole("combobox", { name: "Load a saved context" }), {
      target: { value: savedOption.getAttribute("value") },
    });
    fireEvent.click(screen.getByRole("button", { name: "Load" }));

    expect(screen.getByRole("slider", { name: "Rating for Alien" })).toHaveValue("4.5");
  });

  it("exposes advanced batch controls and stops an auto sweep", async () => {
    render(<App />);
    await loadModel();
    await addContext();
    fireEvent.click(screen.getByRole("tab", { name: /auto/i }));
    fireEvent.change(screen.getByRole("spinbutton", { name: "From year" }), { target: { value: "1990" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Through year" }), { target: { value: "2010" } });
    const popularity = screen.getByRole("combobox", { name: "Minimum popularity" });
    expect(popularity).toHaveValue("0");
    fireEvent.change(popularity, { target: { value: "100000" } });
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(popularity).toHaveValue("0");
    fireEvent.change(screen.getByRole("spinbutton", { name: "From year" }), { target: { value: "1990" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Through year" }), { target: { value: "2010" } });
    fireEvent.change(popularity, { target: { value: "100000" } });
    fireEvent.click(screen.getByText("Genres"));
    fireEvent.click(screen.getByRole("checkbox", { name: "drama" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "science fiction" }));
    fireEvent.click(screen.getByText(/Run settings/));

    const batchField = screen.getByRole("spinbutton", { name: /Batch size/ });
    fireEvent.change(batchField, { target: { value: "8" } });
    fireEvent.click(screen.getByRole("button", { name: /Start sweep/ }));
    const stopButton = await screen.findByRole("button", { name: /Stop sweep/ });
    expect(popularity).toBeDisabled();
    fireEvent.click(stopButton);

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls;
      expect(calls.some(([url, init]) => String(url) === "/api/auto-runs/run-1" && init?.method === "DELETE")).toBe(true);
      const start = calls.find(([url]) => String(url) === "/api/auto-runs");
      expect(JSON.parse(String(start?.[1]?.body))).toMatchObject({
        min_year: 1990,
        max_year: 2010,
        genres: ["drama", "science fiction"],
        min_rating_count: 100000,
      });
    });
    expect(within(screen.getByRole("tabpanel")).getByText(/Ready for continuous sweep|Sweep in progress/)).toBeInTheDocument();
  });
});
