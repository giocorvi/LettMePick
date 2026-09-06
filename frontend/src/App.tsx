import { useEffect, useId, useRef, useState } from "react";
import {
  api,
  type AutoResponse,
  type ContextMovie,
  type LoadedModel,
  type ModelSummary,
  type Movie,
  type MovieFacets,
  type RankedMovie,
  type SavedContext,
} from "./api";

type Mode = "manual" | "auto";
const MAX_CONTEXT_SIZE = 512;

function MovieSearch({
  label,
  disabled,
  excluded,
  onSelect,
}: {
  label: string;
  disabled: boolean;
  excluded: Set<string>;
  onSelect: (movie: Movie) => void;
}) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [movies, setMovies] = useState<Movie[]>([]);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);

  useEffect(() => {
    if (query.trim().length < 1 || disabled) {
      setMovies([]);
      return;
    }
    let current = true;
    const timer = window.setTimeout(() => {
      api.searchMovies(query)
        .then(({ movies: results }) => {
          if (current) {
            setMovies(results.filter((movie) => !excluded.has(movie.id)));
            setOpen(true);
            setActiveIndex(-1);
          }
        })
        .catch(() => current && setMovies([]));
    }, 180);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [disabled, excluded, query]);

  const choose = (movie: Movie) => {
    onSelect(movie);
    setQuery("");
    setMovies([]);
    setOpen(false);
  };

  return (
    <div className="movie-search">
      <label htmlFor={id}>{label}</label>
      <div className="search-field">
        <span aria-hidden="true">⌕</span>
        <input
          id={id}
          role="combobox"
          aria-expanded={open && movies.length > 0}
          aria-controls={`${id}-results`}
          aria-activedescendant={activeIndex >= 0 ? `${id}-option-${activeIndex}` : undefined}
          autoComplete="off"
          disabled={disabled}
          value={query}
          placeholder={disabled ? "Load a model first" : "Type a title…"}
          onChange={(event) => setQuery(event.target.value)}
          onFocus={() => setOpen(true)}
          onKeyDown={(event) => {
            if (!movies.length) return;
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setActiveIndex((index) => Math.min(index + 1, movies.length - 1));
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              setActiveIndex((index) => Math.max(index - 1, 0));
            } else if (event.key === "Enter" && activeIndex >= 0) {
              event.preventDefault();
              choose(movies[activeIndex]);
            } else if (event.key === "Escape") {
              setOpen(false);
            }
          }}
        />
      </div>
      {open && query && (
        <div className="search-results" id={`${id}-results`} role="listbox">
          {movies.length ? (
            movies.map((movie, index) => (
              <button
                id={`${id}-option-${index}`}
                role="option"
                aria-selected={index === activeIndex}
                className={index === activeIndex ? "active" : ""}
                key={movie.id}
                type="button"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => choose(movie)}
              >
                <span>{movie.title}</span>
                <small>{movie.year}</small>
              </button>
            ))
          ) : (
            <p>No matching reels</p>
          )}
        </div>
      )}
    </div>
  );
}

function RatingControl({
  movie,
  onChange,
}: {
  movie: ContextMovie;
  onChange: (rating: number) => void;
}) {
  return (
    <label className="rating-control">
      <span className="sr-only">Rating for {movie.title}</span>
      <span className="star-track" aria-hidden="true">
        <span>★★★★★</span>
        <span className="star-fill" style={{ width: `${movie.rating * 20}%` }}>★★★★★</span>
      </span>
      <input
        aria-label={`Rating for ${movie.title}`}
        type="range"
        min="0.5"
        max="5"
        step="0.5"
        value={movie.rating}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <output>{movie.rating.toFixed(1)}</output>
    </label>
  );
}

function RankingList({ results, visible }: { results: RankedMovie[]; visible: number }) {
  if (!results.length) {
    return (
      <div className="empty-results">
        <div className="empty-orbit" aria-hidden="true"><span /></div>
        <p>The projector is idle.</p>
        <small>Build a context, then send a selection through the model.</small>
      </div>
    );
  }
  return (
    <ol className="ranking-list">
      {results.slice(0, visible).map((movie, index) => (
        <li key={movie.id} style={{ "--rank-delay": `${Math.min(index, 12) * 24}ms` } as React.CSSProperties}>
          <span className="rank">{String(index + 1).padStart(2, "0")}</span>
          <div className="result-copy">
            <h3>{movie.title}</h3>
            <p>{movie.year} <i /> {movie.genres.slice(0, 3).join(" · ") || "Unclassified"}</p>
          </div>
          <div className="predicted-score">
            <span>{movie.score_five.toFixed(2)}</span>
            <small>predicted / 5</small>
          </div>
          <div className="score-line" style={{ "--score": `${movie.score * 100}%` } as React.CSSProperties} />
        </li>
      ))}
    </ol>
  );
}

export default function App() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [loaded, setLoaded] = useState<LoadedModel | null>(null);
  const [context, setContext] = useState<ContextMovie[]>([]);
  const [savedContexts, setSavedContexts] = useState<SavedContext[]>([]);
  const [selectedContextId, setSelectedContextId] = useState("");
  const [contextName, setContextName] = useState("");
  const [contextNotice, setContextNotice] = useState("");
  const [archiveBusy, setArchiveBusy] = useState(false);
  const [query, setQuery] = useState<Movie[]>([]);
  const [mode, setMode] = useState<Mode>("manual");
  const [results, setResults] = useState<RankedMovie[]>([]);
  const [visible, setVisible] = useState(20);
  const [busy, setBusy] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [loadingStage, setLoadingStage] = useState("");
  const [error, setError] = useState("");
  const [configText, setConfigText] = useState<string | null>(null);
  const [batchSize, setBatchSize] = useState(64);
  const [movieFacets, setMovieFacets] = useState<MovieFacets | null>(null);
  const [minYear, setMinYear] = useState<number | null>(null);
  const [maxYear, setMaxYear] = useState<number | null>(null);
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [minRatingCount, setMinRatingCount] = useState(0);
  const [auto, setAuto] = useState<AutoResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [autoTick, setAutoTick] = useState(0);
  const runningRef = useRef(false);

  useEffect(() => {
    api.models()
      .then(({ models: entries }) => {
        setModels(entries);
        if (entries.length) setSelectedModel(entries[0].id);
      })
      .catch((caught: Error) => setError(caught.message));
    api.contexts()
      .then(({ contexts }) => setSavedContexts(contexts))
      .catch((caught: Error) => setError(caught.message));
  }, []);

  useEffect(() => {
    if (!running || !auto || auto.complete) return;
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const next = await api.nextAuto(auto.run_id);
        if (!cancelled && runningRef.current) {
          setAuto(next);
          setResults(next.results);
          if (next.complete) {
            runningRef.current = false;
            setRunning(false);
          } else {
            setAutoTick((tick) => tick + 1);
          }
        }
      } catch (caught) {
        if (!cancelled) {
          runningRef.current = false;
          setRunning(false);
          setError((caught as Error).message);
        }
      }
    }, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [auto, autoTick, running]);

  const activeSummary = models.find((model) => model.id === selectedModel);
  const selectedSavedContext = savedContexts.find((item) => item.id === selectedContextId);
  const contextIds = new Set(context.map((movie) => movie.id));
  const allSelectedIds = new Set([...contextIds, ...query.map((movie) => movie.id)]);
  const yearRangeInvalid = minYear !== null && maxYear !== null && minYear > maxYear;

  const resetExperiment = () => {
    setContext([]);
    setQuery([]);
    setResults([]);
    setVisible(20);
    setAuto(null);
    setContextNotice("");
    setRunning(false);
    runningRef.current = false;
  };

  const saveContext = async () => {
    const name = contextName.trim();
    if (!loaded || !name || !context.length) return;
    const existing = savedContexts.find(
      (item) => item.model_id === loaded.id && item.name.toLocaleLowerCase() === name.toLocaleLowerCase(),
    );
    setArchiveBusy(true);
    setError("");
    try {
      const saved = await api.saveContext(name, loaded.id, context);
      setSavedContexts((current) => [saved, ...current.filter((item) => item.id !== saved.id)]);
      setSelectedContextId(saved.id);
      setContextName("");
      setContextNotice(`${existing ? "Updated" : "Saved"} “${name}” on this machine.`);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setArchiveBusy(false);
    }
  };

  const loadSavedContext = () => {
    if (!loaded || !selectedSavedContext) return;
    if (selectedSavedContext.model_id !== loaded.id) {
      setError(`Load ${selectedSavedContext.model_name} before using this context.`);
      return;
    }
    const movies = selectedSavedContext.movies.map((movie) => ({
      ...movie,
      genres: [...movie.genres],
    }));
    const ids = new Set(movies.map((movie) => movie.id));
    setContext(movies);
    setQuery((current) => current.filter((movie) => !ids.has(movie.id)));
    setResults([]);
    setVisible(20);
    setAuto(null);
    setContextNotice(`Loaded “${selectedSavedContext.name}” — ${movies.length} rated films.`);
  };

  const deleteSavedContext = async () => {
    if (!selectedSavedContext) return;
    setArchiveBusy(true);
    setError("");
    try {
      await api.deleteContext(selectedSavedContext.id);
      setSavedContexts((current) => current.filter((item) => item.id !== selectedSavedContext.id));
      setSelectedContextId("");
      setContextNotice(`Deleted “${selectedSavedContext.name}”.`);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setArchiveBusy(false);
    }
  };

  const loadModel = async () => {
    if (!selectedModel) return;
    setBusy(true);
    setLoadingModel(true);
    setLoadingStage("Starting load");
    setError("");
    if (auto?.run_id) await api.stopAuto(auto.run_id).catch(() => undefined);
    const progressTimer = window.setInterval(() => {
      void api.models().then((response) => {
        const loading = response.models.find((model) => model.id === selectedModel);
        if (loading?.loading_stage) setLoadingStage(loading.loading_stage);
      }).catch(() => undefined);
    }, 750);
    try {
      const model = await api.loadModel(selectedModel);
      const facets = await api.movieFacets();
      setLoaded(model);
      setMovieFacets(facets);
      setMinYear(facets.min_year);
      setMaxYear(facets.max_year);
      setSelectedGenres([]);
      setMinRatingCount(0);
      resetExperiment();
      const refreshed = await api.models();
      setModels(refreshed.models);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      window.clearInterval(progressTimer);
      setLoadingModel(false);
      setLoadingStage("");
      setBusy(false);
    }
  };

  const inspectConfig = async () => {
    if (!activeSummary?.has_config) return;
    setError("");
    try {
      setConfigText((await api.inspectConfig(activeSummary.id)).content);
    } catch (caught) {
      setError((caught as Error).message);
    }
  };

  const runManual = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await api.manual(context, query);
      setResults(response.results);
      setVisible(20);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const startAuto = async () => {
    setBusy(true);
    setError("");
    setResults([]);
    setVisible(20);
    try {
      const response = await api.startAuto(context, batchSize, {
        minYear,
        maxYear,
        genres: selectedGenres,
        minRatingCount,
      });
      setAuto(response);
      setResults(response.results);
      setRunning(!response.complete);
      runningRef.current = !response.complete;
      setAutoTick((tick) => tick + 1);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const stopAuto = async () => {
    runningRef.current = false;
    setRunning(false);
    if (auto?.run_id) await api.stopAuto(auto.run_id).catch(() => undefined);
  };

  return (
    <div className="app-shell">
      <div className="grain" aria-hidden="true" />
      <header className="masthead">
        <div className="wordmark" aria-label="LettMePick screening room home">
          <span>Lett</span><em>Me</em><span>Pick</span>
        </div>
        <div className="room-title">
          <span>MODEL TESTING WORKBENCH</span>
          <p>Screening Room № 01</p>
        </div>
        <div className={`status-lamp ${loaded ? "online" : ""}`}>
          <i aria-hidden="true" /> {loaded ? `${loaded.device} online` : "standby"}
        </div>
      </header>

      {error && (
        <div className="error-strip" role="alert">
          <span>Projection fault</span><p>{error}</p>
          <button type="button" onClick={() => setError("")} aria-label="Dismiss error">×</button>
        </div>
      )}

      <main id="top" className="workbench">
        <aside className="context-panel">
          <section className="model-rack panel-block">
            <div className="section-label"><span>01</span><p>Choose checkpoint</p></div>
            {models.length ? (
              <>
                <select
                  aria-label="Registered model"
                  value={selectedModel}
                  disabled={busy || running}
                  onChange={(event) => setSelectedModel(event.target.value)}
                >
                  {models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
                </select>
                <p className="model-description">{activeSummary?.description || "No production notes attached to this reel."}</p>
                <div className="model-actions">
                  <button className="button primary compact" type="button" disabled={busy || running} onClick={loadModel}>
                    {loadingModel ? "Loading model…" : loaded?.id === selectedModel ? "Reload model" : "Load model"}
                  </button>
                  {activeSummary?.has_config && (
                    <button className="text-button" type="button" onClick={inspectConfig}>Inspect config</button>
                  )}
                </div>
                {loadingModel && <p className="load-stage" role="status">{loadingStage}…</p>}
              </>
            ) : (
              <div className="no-models"><p>No reels registered</p><code>configs/models.toml</code></div>
            )}
          </section>

          <section className="context-builder panel-block">
            <div className="section-label"><span>02</span><p>Build viewer context</p><b>{context.length}/{MAX_CONTEXT_SIZE}</b></div>
            <MovieSearch
              label="Find a rated movie"
              disabled={!loaded || running || context.length >= MAX_CONTEXT_SIZE}
              excluded={contextIds}
              onSelect={(movie) => setContext((current) => (
                current.length < MAX_CONTEXT_SIZE
                  ? [...current, { ...movie, rating: 3.5 }]
                  : current
              ))}
            />
            <details className="context-archive">
              <summary>
                <span>Context archive</span>
                <b>{savedContexts.length} saved</b>
              </summary>
              <div className="archive-controls">
                <div className="archive-save">
                  <label htmlFor="context-name">Save current context</label>
                  <div>
                    <input
                      id="context-name"
                      type="text"
                      maxLength={60}
                      value={contextName}
                      disabled={!loaded || running}
                      placeholder="e.g. Sunday matinee"
                      onChange={(event) => setContextName(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          saveContext();
                        }
                      }}
                    />
                    <button type="button" disabled={!loaded || !context.length || !contextName.trim() || running || archiveBusy} onClick={saveContext}>{archiveBusy ? "Working…" : "Save"}</button>
                  </div>
                </div>
                <div className="archive-load">
                  <label htmlFor="saved-context">Load a saved context</label>
                  <select
                    id="saved-context"
                    value={selectedContextId}
                    disabled={!savedContexts.length || running || archiveBusy}
                    onChange={(event) => setSelectedContextId(event.target.value)}
                  >
                    <option value="">Select a context…</option>
                    {savedContexts.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name} · {item.movies.length} films{loaded && item.model_id !== loaded.id ? ` · ${item.model_name}` : ""}
                      </option>
                    ))}
                  </select>
                  <div>
                    <button type="button" disabled={!selectedSavedContext || !loaded || selectedSavedContext.model_id !== loaded.id || running || archiveBusy} onClick={loadSavedContext}>Load</button>
                    <button className="archive-delete" type="button" disabled={!selectedSavedContext || running || archiveBusy} onClick={deleteSavedContext}>Delete</button>
                  </div>
                </div>
                {contextNotice && <p className="archive-notice" role="status">{contextNotice}</p>}
              </div>
            </details>
            <div className="context-list" aria-live="polite">
              {context.map((movie) => (
                <article key={movie.id}>
                  <div><h3>{movie.title}</h3><p>{movie.year} · {movie.genres[0] || "film"}</p></div>
                  <RatingControl
                    movie={movie}
                    onChange={(rating) => setContext((current) => current.map((item) => item.id === movie.id ? { ...item, rating } : item))}
                  />
                  <button
                    className="remove"
                    aria-label={`Remove ${movie.title}`}
                    type="button"
                    disabled={running}
                    onClick={() => setContext((current) => current.filter((item) => item.id !== movie.id))}
                  >×</button>
                </article>
              ))}
              {!context.length && <p className="context-empty">Your taste profile begins with one film.</p>}
            </div>
          </section>
        </aside>

        <section className="screening-panel">
          <div className="mode-header">
            <div className="eyebrow">03 / RUN THE RECOMMENDER</div>
            <div className="mode-switch" role="tablist" aria-label="Inference mode">
              {(["manual", "auto"] as Mode[]).map((item) => (
                <button
                  key={item}
                  role="tab"
                  aria-selected={mode === item}
                  type="button"
                  disabled={running}
                  onClick={() => { setMode(item); setResults([]); setVisible(20); }}
                >
                  {item}<span>{item === "manual" ? "Curated selection" : "Continuous sweep"}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="control-stage">
            {mode === "manual" ? (
              <div className="manual-controls" role="tabpanel">
                <MovieSearch
                  label="Add movies to the query slate"
                  disabled={!loaded || running}
                  excluded={allSelectedIds}
                  onSelect={(movie) => setQuery((current) => [...current, movie])}
                />
                <div className="query-slate">
                  {query.map((movie, index) => (
                    <button type="button" key={movie.id} onClick={() => setQuery((current) => current.filter(({ id }) => id !== movie.id))}>
                      <span>{String(index + 1).padStart(2, "0")}</span>{movie.title}<i>×</i>
                    </button>
                  ))}
                  {!query.length && <p>Select the films you want this model to judge.</p>}
                </div>
                <div className="run-bar">
                  <p><strong>{query.length}</strong> query reels</p>
                  <button className="button primary" type="button" disabled={busy || !loaded || !context.length || !query.length} onClick={runManual}>
                    {busy ? "Scoring…" : "Score the slate"}<span>→</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="auto-controls" role="tabpanel">
                <div className="auto-copy">
                  <span className={`pulse ${running ? "live" : ""}`} aria-hidden="true" />
                  <div><h2>{running ? "Sweep in progress" : auto?.complete ? "Catalog exhausted" : "Ready for continuous sweep"}</h2>
                    <p>Each pass samples unseen films and keeps the strongest predictions on top.</p></div>
                </div>
                <section className="auto-filters" aria-labelledby="candidate-filters-title">
                  <header>
                    <div>
                      <span>Candidate pool</span>
                      <h3 id="candidate-filters-title">Filter the sweep</h3>
                    </div>
                    <button
                      type="button"
                      disabled={!movieFacets || running}
                      onClick={() => {
                        setMinYear(movieFacets?.min_year ?? null);
                        setMaxYear(movieFacets?.max_year ?? null);
                        setSelectedGenres([]);
                        setMinRatingCount(0);
                      }}
                    >Reset</button>
                  </header>
                  <div className="year-filter">
                    <label>From year
                      <input
                        aria-invalid={yearRangeInvalid}
                        type="number"
                        min={movieFacets?.min_year}
                        max={movieFacets?.max_year}
                        value={minYear ?? ""}
                        disabled={!movieFacets || running}
                        onChange={(event) => setMinYear(event.target.value ? Number(event.target.value) : null)}
                      />
                    </label>
                    <i aria-hidden="true" />
                    <label>Through year
                      <input
                        aria-invalid={yearRangeInvalid}
                        type="number"
                        min={movieFacets?.min_year}
                        max={movieFacets?.max_year}
                        value={maxYear ?? ""}
                        disabled={!movieFacets || running}
                        onChange={(event) => setMaxYear(event.target.value ? Number(event.target.value) : null)}
                      />
                    </label>
                  </div>
                  {yearRangeInvalid && <p className="filter-error" role="alert">The first year must not exceed the last.</p>}
                  <label className="popularity-filter">
                    <span>Minimum popularity</span>
                    <select
                      aria-label="Minimum popularity"
                      value={minRatingCount}
                      disabled={!movieFacets || running}
                      onChange={(event) => setMinRatingCount(Number(event.target.value))}
                    >
                      {movieFacets?.popularity_levels.map((level) => (
                        <option key={level.id} value={level.min_rating_count}>
                          {level.label} · {level.min_rating_count === 0
                            ? "no minimum"
                            : `${level.min_rating_count.toLocaleString()}+ ratings`}
                        </option>
                      ))}
                    </select>
                  </label>
                  <details className="genre-filter">
                    <summary>
                      <span>Genres</span>
                      <b>{selectedGenres.length ? `${selectedGenres.length} selected · match any` : "All genres"}</b>
                    </summary>
                    <p>A movie passes when it overlaps any selected genre.</p>
                    <div>
                      {movieFacets?.genres.map((genre) => (
                        <label key={genre}>
                          <input
                            type="checkbox"
                            checked={selectedGenres.includes(genre)}
                            disabled={running}
                            onChange={(event) => setSelectedGenres((current) => (
                              event.target.checked
                                ? [...current, genre]
                                : current.filter((item) => item !== genre)
                            ))}
                          />
                          <span>{genre}</span>
                        </label>
                      ))}
                    </div>
                  </details>
                </section>
                <details className="run-settings">
                  <summary>Run settings <span>advanced</span></summary>
                  <label>Batch size
                    <input type="number" min="1" max="512" value={batchSize} disabled={running} onChange={(event) => setBatchSize(Math.max(1, Math.min(512, Number(event.target.value))))} />
                  </label>
                </details>
                <div className="run-bar">
                  <div className="run-metrics">
                    <p><strong>{auto?.batches || 0}</strong><span>batches</span></p>
                    <p><strong>{auto?.sampled || 0}</strong><span>sampled</span></p>
                    <p><strong>{auto?.remaining ?? loaded?.catalog_size ?? 0}</strong><span>remaining</span></p>
                  </div>
                  {running ? (
                    <button className="button stop" type="button" onClick={stopAuto}>Stop sweep <span>■</span></button>
                  ) : (
                    <button className="button primary" type="button" disabled={busy || !loaded || !context.length || yearRangeInvalid} onClick={startAuto}>
                      {busy ? "Starting…" : auto ? "Restart sweep" : "Start sweep"}<span>→</span>
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="results-header">
            <div><span>RANKED OUTPUT</span><h1>Tonight’s programme</h1></div>
            <p>{results.length ? `${results.length} films scored` : "Awaiting inference"}</p>
          </div>
          <RankingList results={results} visible={visible} />
          {visible < results.length && (
            <button className="load-more" type="button" onClick={() => setVisible((count) => count + 20)}>
              Load next 20 <span>{visible} / {results.length}</span>
            </button>
          )}
        </section>
      </main>

      <footer><span>LET ME PICK / EXPERIMENTAL PICTURE HOUSE</span><p>Predictions are model estimates, not audience ratings.</p></footer>

      {configText !== null && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setConfigText(null)}>
          <section className="config-modal" role="dialog" aria-modal="true" aria-labelledby="config-title" onMouseDown={(event) => event.stopPropagation()}>
            <header><div><span>REGISTRY OVERRIDE</span><h2 id="config-title">Model configuration</h2></div><button aria-label="Close config" type="button" onClick={() => setConfigText(null)}>×</button></header>
            <p className="inspection-warning"><i /> This configuration overrides values embedded in the checkpoint.</p>
            <pre>{configText}</pre>
          </section>
        </div>
      )}
    </div>
  );
}
