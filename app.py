"""
Movie Analyzer – Streamlit Web Application
Search any movie in the dataset, or paste a new synopsis to get a full analysis report.
"""

import re
from pathlib import Path
from difflib import get_close_matches

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("models/movie_analyzer.joblib")
SCRIPTS_DIR = Path("Movie scripts/scirpt")

# ── sentiment / rating colour helpers ────────────────────────────────────────
SENTIMENT_COLOR = {
    "positive": "#22c55e",
    "neutral":  "#f59e0b",
    "negative": "#ef4444",
}
SENTIMENT_ICON = {
    "positive": "😊",
    "neutral":  "😐",
    "negative": "😞",
}

GOOD_COLOR = "#22c55e"
BAD_COLOR  = "#ef4444"


def rating_color(rating: float) -> str:
    if rating >= 7.5:
        return "#22c55e"
    if rating >= 6.0:
        return "#f59e0b"
    return "#ef4444"


# ── model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ── prediction helpers ────────────────────────────────────────────────────────
def normalize_key(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    return text.strip("_")


def read_script(path: Path, max_chars: int = 25_000) -> str:
    content = path.read_text(encoding="utf-8", errors="ignore")
    return re.sub(r"\s+", " ", content).strip()[:max_chars]


def run_prediction(artifact, movie_title: str, text: str):
    cleaned = re.sub(r"\s+", " ", text).strip()
    model_text = f"{movie_title.strip()} . {cleaned}"
    vec = artifact["vectorizer"].transform([model_text])

    genre_prob  = artifact["genre_model"].predict_proba(vec)[0]
    sent_prob   = artifact["sentiment_model"].predict_proba(vec)[0]
    gb_prob     = artifact["good_bad_model"].predict_proba(vec)[0]
    pred_rating = float(np.clip(artifact["rating_model"].predict(vec)[0], 0.0, 10.0))

    sent_classes = artifact["sentiment_model"].classes_
    gb_classes   = artifact["good_bad_model"].classes_
    sent_label   = str(sent_classes[int(np.argmax(sent_prob))])

    good_idx  = list(gb_classes).index("good") if "good" in gb_classes else int(np.argmax(gb_prob))
    prob_good = float(gb_prob[good_idx])
    gb_label  = "good" if prob_good >= 0.5 else "bad"

    # genres sorted by probability
    genre_pairs = sorted(
        zip(artifact["mlb"].classes_, genre_prob.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    top_genres = [(g, p) for g, p in genre_pairs if p >= 0.35]
    if not top_genres:
        top_genres = [genre_pairs[0]]
    top_genres = top_genres[:6]

    # similar movies
    distances, indices = artifact["neighbors"].kneighbors(vec, n_neighbors=min(8, len(artifact["movies"])))
    norm_input = normalize_key(movie_title)
    similar = []
    for dist, idx in zip(distances[0], indices[0]):
        title = artifact["movies"][idx]
        if normalize_key(title) == norm_input:
            continue
        similar.append({
            "movie":         title,
            "similarity":    round(float(1.0 - dist), 4),
            "known_rating":  round(float(artifact["ratings"][idx]), 1),
            "known_genres":  artifact["genres"][idx],
        })
        if len(similar) >= 5:
            break

    return {
        "title":       movie_title,
        "genres":      top_genres,
        "all_genres":  genre_pairs,
        "sentiment":   sent_label,
        "sent_prob":   {str(c): round(float(p), 4) for c, p in zip(sent_classes, sent_prob)},
        "good_bad":    gb_label,
        "prob_good":   round(prob_good, 4),
        "rating":      round(pred_rating, 2),
        "similar":     similar,
    }


# ── chart builders ────────────────────────────────────────────────────────────
def genre_bar_chart(genre_pairs):
    names  = [g for g, _ in genre_pairs[:10]]
    values = [round(p * 100, 1) for _, p in genre_pairs[:10]]
    colors = ["#6366f1" if v >= 35 else "#94a3b8" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x}%<extra></extra>",
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=50, t=10, b=10),
        xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    return fig


def sentiment_donut(sent_prob):
    labels = list(sent_prob.keys())
    values = [v * 100 for v in sent_prob.values()]
    colors = [SENTIMENT_COLOR.get(l, "#94a3b8") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    return fig


def rating_gauge(rating: float):
    color = rating_color(rating)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rating,
        number={"suffix": " / 10", "font": {"size": 28, "color": color}},
        gauge={
            "axis":  {"range": [0, 10], "tickwidth": 1, "tickcolor": "#475569"},
            "bar":   {"color": color, "thickness": 0.28},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 5.5],  "color": "#1e293b"},
                {"range": [5.5, 7],  "color": "#1e293b"},
                {"range": [7, 10],   "color": "#1e293b"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": rating,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    return fig


# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f172a;
    color: #e2e8f0;
}
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: #f1f5f9; }
.report-card {
    background: #1e293b;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 14px;
}
.metric-box {
    background: #0f172a;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px 3px;
}
.badge-genre  { background:#312e81; color:#c7d2fe; }
.badge-good   { background:#14532d; color:#86efac; }
.badge-bad    { background:#7f1d1d; color:#fca5a5; }
.similar-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 6px;
    border-bottom: 1px solid #334155;
}
.sim-title { font-weight: 600; color: #e2e8f0; }
.sim-meta  { font-size: 0.8rem; color: #94a3b8; margin-top: 2px; }
.divider   { border: none; border-top: 1px solid #334155; margin: 10px 0; }
</style>
"""


# ── main app ──────────────────────────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # header
    st.markdown("## 🎬 Movie Analyzer")
    st.markdown(
        "<p style='color:#94a3b8;margin-top:-10px'>Search any movie to get genre detection, "
        "sentiment analysis, quality verdict, predicted rating and similar films.</p>",
        unsafe_allow_html=True,
    )

    artifact = load_artifact()
    if artifact is None:
        st.error("⚠️ Model not found. Run `python movie_analyzer.py train --base-dir .` first.")
        return

    known_titles: list[str] = artifact["movies"]
    known_lower  = {t.lower().strip(): t for t in known_titles}

    # ── search bar ────────────────────────────────────────────────────────────
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    col_search, col_btn = st.columns([5, 1])
    with col_search:
        query = st.text_input(
            "Search movie title",
            placeholder="e.g.  Interstellar, The Room, Deadpool …",
            label_visibility="collapsed",
        )
    with col_btn:
        search_hit = st.button("🔍  Analyze", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not query.strip():
        st.info("Type a movie title above to get started.", icon="🎥")
        return

    # ── resolve movie ─────────────────────────────────────────────────────────
    q_lower   = query.strip().lower()
    exact_key = known_lower.get(q_lower)

    # fuzzy suggestion if no exact hit
    suggestions = get_close_matches(q_lower, list(known_lower.keys()), n=3, cutoff=0.45)
    canonical_title = known_lower.get(suggestions[0]) if suggestions else query.strip()
    if exact_key:
        canonical_title = exact_key

    is_known = canonical_title in known_titles

    # ── if known, show suggestion note; locate its script file ───────────────
    if not exact_key and suggestions:
        chosen = st.selectbox(
            "Did you mean…",
            [known_lower[s] for s in suggestions] + ["Enter custom synopsis →"],
            index=0,
        )
        if chosen != "Enter custom synopsis →":
            canonical_title = chosen
            is_known = True

    # ── get text for prediction ────────────────────────────────────────────────
    script_text = ""
    script_found = False

    if is_known:
        idx = known_titles.index(canonical_title)
        script_path = artifact["script_paths"][idx] if idx < len(artifact["script_paths"]) else ""
        if script_path and Path(script_path).exists():
            script_text  = read_script(Path(script_path))
            script_found = True

    if not script_found:
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.markdown("##### ✍️ No synopsis found — paste one below")
        synopsis_input = st.text_area(
            "Synopsis / Script excerpt",
            height=150,
            placeholder="Paste a brief synopsis or any plot description here…",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        script_text = synopsis_input.strip()
        if not script_text:
            st.warning("Paste a synopsis above and click **Analyze** to continue.", icon="📝")
            return

    if not search_hit and not script_found:
        return

    # ── run prediction ────────────────────────────────────────────────────────
    with st.spinner("Analyzing…"):
        result = run_prediction(artifact, canonical_title, script_text)

    # ══════════════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════════════

    # ── title row ─────────────────────────────────────────────────────────────
    gb_color  = GOOD_COLOR if result["good_bad"] == "good" else BAD_COLOR
    gb_class  = "badge-good" if result["good_bad"] == "good" else "badge-bad"
    gb_icon   = "✅" if result["good_bad"] == "good" else "❌"
    sent_icon = SENTIMENT_ICON.get(result["sentiment"], "")
    sent_col  = SENTIMENT_COLOR.get(result["sentiment"], "#94a3b8")

    genre_badges = "".join(
        f"<span class='badge badge-genre'>{g}</span>"
        for g, _ in result["genres"]
    )

    st.markdown(f"""
    <div class='report-card'>
        <h2 style='margin:0 0 6px 0'>{result['title']}</h2>
        <div style='margin-bottom:10px'>{genre_badges}</div>
        <div style='display:flex;gap:20px;flex-wrap:wrap;align-items:center'>
            <span class='badge {gb_class}'>{gb_icon} {result['good_bad'].upper()}</span>
            <span style='color:{sent_col};font-weight:600;font-size:1rem'>{sent_icon} {result['sentiment'].capitalize()}</span>
            <span style='color:{rating_color(result['rating'])};font-weight:700;font-size:1.1rem'>
                ⭐ {result['rating']} / 10
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── three metric panels ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='report-card'><b>📊 Predicted Rating</b></div>", unsafe_allow_html=True)
        st.plotly_chart(rating_gauge(result["rating"]), use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("<div class='report-card'><b>🎭 Sentiment Breakdown</b></div>", unsafe_allow_html=True)
        st.plotly_chart(sentiment_donut(result["sent_prob"]), use_container_width=True, config={"displayModeBar": False})

    with col3:
        st.markdown("<div class='report-card'><b>🏷️ Genre Probabilities</b></div>", unsafe_allow_html=True)
        st.plotly_chart(genre_bar_chart(result["all_genres"]), use_container_width=True, config={"displayModeBar": False})

    # ── detailed stats + similar movies ───────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.markdown("**📋 Analysis Summary**")

        prob_good_pct = round(result["prob_good"] * 100, 1)
        prob_bad_pct  = round((1 - result["prob_good"]) * 100, 1)
        sent_confidence = round(max(result["sent_prob"].values()) * 100, 1)
        top_genre_name, top_genre_prob = result["genres"][0]

        rows = [
            ("Verdict",            f"{gb_icon} {result['good_bad'].upper()}  ({prob_good_pct}% good / {prob_bad_pct}% bad)"),
            ("Sentiment",          f"{sent_icon} {result['sentiment'].capitalize()}  ({sent_confidence}% confidence)"),
            ("Predicted Rating",   f"⭐ {result['rating']} / 10"),
            ("Top Genre",          f"{top_genre_name.title()}  ({round(top_genre_prob*100, 1)}%)"),
            ("All Predicted Genres", ", ".join(g.title() for g, _ in result["genres"])),
        ]
        for label, value in rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:8px 0;"
                f"border-bottom:1px solid #334155'>"
                f"<span style='color:#94a3b8'>{label}</span>"
                f"<span style='font-weight:600;text-align:right;max-width:65%'>{value}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.markdown("**🎥 Similar Movies**")
        if result["similar"]:
            for movie in result["similar"]:
                sim_pct   = round(movie["similarity"] * 100, 1)
                r_color   = rating_color(movie["known_rating"])
                genre_str = ", ".join(g.title() for g in movie["known_genres"][:3])
                st.markdown(
                    f"<div class='similar-row'>"
                    f"  <div>"
                    f"    <div class='sim-title'>{movie['movie']}</div>"
                    f"    <div class='sim-meta'>{genre_str}</div>"
                    f"  </div>"
                    f"  <div style='text-align:right'>"
                    f"    <div style='color:{r_color};font-weight:700'>⭐ {movie['known_rating']}</div>"
                    f"    <div style='color:#64748b;font-size:0.78rem'>{sim_pct}% similar</div>"
                    f"  </div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No similar movies found in dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── genre confidence table (expandable) ───────────────────────────────────
    with st.expander("🔬 Full Genre Confidence Scores"):
        for genre, prob in result["all_genres"]:
            pct = round(prob * 100, 1)
            bar_color = "#6366f1" if pct >= 35 else "#334155"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin:5px 0'>"
                f"  <span style='width:140px;color:#cbd5e1'>{genre.title()}</span>"
                f"  <div style='flex:1;background:#1e293b;border-radius:4px;height:12px'>"
                f"    <div style='width:{pct}%;background:{bar_color};height:100%;border-radius:4px'></div>"
                f"  </div>"
                f"  <span style='width:46px;text-align:right;color:#94a3b8;font-size:0.82rem'>{pct}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
