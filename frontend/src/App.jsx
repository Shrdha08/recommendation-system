import { useEffect, useState } from "react";
import "./App.css";

// Empty string = same-origin (Docker: nginx proxies /api to the backend).
// Unset = local dev against a separately-run backend.
const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function MovieList({ items, emptyText, renderMeta }) {
  if (!items.length) return <p className="empty">{emptyText}</p>;
  return (
    <ul className="movie-list">
      {items.map((m) => (
        <li key={m.movie_idx} className="movie-item">
          <div className="movie-main">
            <span className="movie-title">{m.title}</span>
            <span className="movie-genres">{m.genres.split("|").join(" · ")}</span>
          </div>
          {renderMeta && <span className="movie-meta">{renderMeta(m)}</span>}
        </li>
      ))}
    </ul>
  );
}

export default function App() {
  const [summary, setSummary] = useState(null);
  const [userId, setUserId] = useState(42);
  const [inputId, setInputId] = useState("42");
  const [profile, setProfile] = useState([]);
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/summary`)
      .then((r) => r.json())
      .then(setSummary)
      .catch(() => setError("Cannot reach the API. Is the backend running?"));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      fetch(`${API}/api/users/${userId}/profile?limit=10`).then((r) => {
        if (!r.ok) throw new Error("User not found");
        return r.json();
      }),
      fetch(`${API}/api/users/${userId}/recommendations?top_k=10`).then((r) => {
        if (!r.ok) throw new Error("User not found");
        return r.json();
      }),
    ])
      .then(([p, r]) => {
        if (cancelled) return;
        setProfile(p);
        setRecs(r);
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));

    return () => {
      cancelled = true;
    };
  }, [userId]);

  const submit = (e) => {
    e.preventDefault();
    const n = parseInt(inputId, 10);
    if (!Number.isNaN(n)) setUserId(n);
  };

  const maxUser = summary ? summary.user_count - 1 : 3373;

  return (
    <div className="page">
      <header>
        <h1>Movie Recommendations</h1>
        <p className="subtitle">
          Neural Collaborative Filtering (NeuMF)
          {summary && ` · ${summary.user_count} users · ${summary.movie_count} movies`}
        </p>
      </header>

      <form className="controls" onSubmit={submit}>
        <label htmlFor="user">User ID</label>
        <input
          id="user"
          type="number"
          min="0"
          max={maxUser}
          value={inputId}
          onChange={(e) => setInputId(e.target.value)}
        />
        <button type="submit">Get recommendations</button>
        <span className="hint">0 – {maxUser}</span>
      </form>

      {error && <p className="error">{error}</p>}
      {loading && <p className="loading">Loading…</p>}

      {!loading && !error && (
        <div className="columns">
          <section>
            <h2>Top rated by this user</h2>
            <MovieList
              items={profile}
              emptyText="No ratings for this user."
              renderMeta={(m) => `${m.rating.toFixed(1)} ★`}
            />
          </section>

          <section>
            <h2>Recommended for this user</h2>
            <MovieList
              items={recs}
              emptyText="No recommendations available."
              renderMeta={(m) => m.predicted_rating.toFixed(2)}
            />
          </section>
        </div>
      )}
    </div>
  );
}
