from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import plotly.graph_objects as go
import streamlit as st

from movie_analyzer import get_known_movie_text, run_prediction


st.set_page_config(
    page_title="Movie Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("models/movie_analyzer.joblib")

SENTIMENT_COLOR = {
    "positive": "#22c55e",
    "neutral": "#f59e0b",
    "negative": "#ef4444",
}
SENTIMENT_ICON = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😞",
}
GOOD_COLOR = "#22c55e"
BAD_COLOR = "#ef4444"

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
.mode-pill {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background: #172554;
    color: #bfdbfe;
    font-size: 0.8rem;
    font-weight: 600;
}
</style>
"""


def rating_color(rating: float) -> str:
    if rating >= 7.5:
        return "#22c55e"
    if rating >= 6.0:
        return "#f59e0b"
    return "#ef4444"


def train_artifact_if_missing() -> Tuple[bool, Optional[str]]:
    if MODEL_PATH.exists():
        return True, None

    project_root = Path(__file__).resolve().parent
    command = [sys.executable, "movie_analyzer.py", "train", "--base-dir", "."]

    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, f"Auto-training failed to start: {exc}"

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or "Unknown training error"
        return False, f"Auto-training failed: {details}"

    if not MODEL_PATH.exists():
        return False, "Auto-training completed but model file was not created."

    return True, None


@st.cache_resource(show_spinner="Loading local model…")
def load_local_artifact() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    ready, error_message = train_artifact_if_missing()
    if not ready:
        return None, error_message

    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, f"Failed to load model artifact: {exc}"


def resolve_runtime() -> Dict[str, Any]:
    artifact, load_error = load_local_artifact()
    if artifact is None:
        return {
            "mode": "none",
            "error": (
                load_error
                or "Local model artifact not found. Run `python movie_analyzer.py train --base-dir .` first."
            ),
        }

    return {
        "mode": "local",
        "artifact": artifact,
        "titles": artifact["movies"],
    }


def format_prediction_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": raw_result["movie"],
        "genres": [
            (item["genre"], float(item["probability"]))
            for item in raw_result["predicted_genres"]
        ],
        "all_genres": [
            (item["genre"], float(item["probability"]))
            for item in raw_result["all_genre_probabilities"]
        ],
        "sentiment": raw_result["predicted_sentiment"]["label"],
        "sent_prob": {
            str(label): float(probability)
            for label, probability in raw_result["predicted_sentiment"]["class_probabilities"].items()
        },
        "good_bad": raw_result["good_or_bad"]["label"],
        "prob_good": float(raw_result["good_or_bad"]["probability_good"]),
        "rating": float(raw_result["predicted_rating"]),
        "similar": raw_result["similar_movies"],
    }


def local_prediction(artifact: Dict[str, Any], movie_title: str, text: str) -> Dict[str, Any]:
    return run_prediction(
        artifact=artifact,
        text=text,
        movie_title=movie_title,
        genre_threshold=0.35,
        top_k_genres=6,
        similar_k=5,
    )


def genre_bar_chart(genre_pairs: List[Tuple[str, float]]):
    names = [genre for genre, _ in genre_pairs[:10]]
    values = [round(probability * 100, 1) for _, probability in genre_pairs[:10]]
    colors = ["#6366f1" if value >= 35 else "#94a3b8" for value in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{value}%" for value in values],
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


def sentiment_donut(sent_prob: Dict[str, float]):
    labels = list(sent_prob.keys())
    values = [value * 100 for value in sent_prob.values()]
    colors = [SENTIMENT_COLOR.get(label, "#94a3b8") for label in labels]

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
            "axis": {"range": [0, 10], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 5.5], "color": "#1e293b"},
                {"range": [5.5, 7], "color": "#1e293b"},
                {"range": [7, 10], "color": "#1e293b"},
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


def render_report(result: Dict[str, Any]) -> None:
    gb_class = "badge-good" if result["good_bad"] == "good" else "badge-bad"
    gb_icon = "✅" if result["good_bad"] == "good" else "❌"
    sent_icon = SENTIMENT_ICON.get(result["sentiment"], "")
    sent_color = SENTIMENT_COLOR.get(result["sentiment"], "#94a3b8")

    genre_badges = "".join(
        f"<span class='badge badge-genre'>{genre}</span>"
        for genre, _ in result["genres"]
    )

    st.markdown(
        f"""
        <div class='report-card'>
            <h2 style='margin:0 0 6px 0'>{result['title']}</h2>
            <div style='margin-bottom:10px'>{genre_badges}</div>
            <div style='display:flex;gap:20px;flex-wrap:wrap;align-items:center'>
                <span class='badge {gb_class}'>{gb_icon} {result['good_bad'].upper()}</span>
                <span style='color:{sent_color};font-weight:600;font-size:1rem'>
                    {sent_icon} {result['sentiment'].capitalize()}
                </span>
                <span style='color:{rating_color(result['rating'])};font-weight:700;font-size:1.1rem'>
                    ⭐ {result['rating']} / 10
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.markdown("**📋 Analysis Summary**")
        prob_good_pct = round(result["prob_good"] * 100, 1)
        prob_bad_pct = round((1 - result["prob_good"]) * 100, 1)
        sentiment_confidence = round(max(result["sent_prob"].values()) * 100, 1)
        top_genre_name, top_genre_prob = result["genres"][0]
        rows = [
            ("Verdict", f"{gb_icon} {result['good_bad'].upper()} ({prob_good_pct}% good / {prob_bad_pct}% bad)"),
            ("Sentiment", f"{sent_icon} {result['sentiment'].capitalize()} ({sentiment_confidence}% confidence)"),
            ("Predicted Rating", f"⭐ {result['rating']} / 10"),
            ("Top Genre", f"{top_genre_name.title()} ({round(top_genre_prob * 100, 1)}%)"),
            ("All Predicted Genres", ", ".join(genre.title() for genre, _ in result["genres"])),
        ]
        for label, value in rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #334155'>"
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
                similarity_pct = round(float(movie["similarity"]) * 100, 1)
                known_rating = float(movie["known_rating"])
                genres = movie.get("known_genres", [])
                genre_text = ", ".join(str(genre).title() for genre in genres[:3])
                st.markdown(
                    f"<div class='similar-row'>"
                    f"<div>"
                    f"<div class='sim-title'>{movie['movie']}</div>"
                    f"<div class='sim-meta'>{genre_text}</div>"
                    f"</div>"
                    f"<div style='text-align:right'>"
                    f"<div style='color:{rating_color(known_rating)};font-weight:700'>⭐ {known_rating}</div>"
                    f"<div style='color:#64748b;font-size:0.78rem'>{similarity_pct}% similar</div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No similar movies found in dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔬 Full Genre Confidence Scores"):
        for genre, probability in result["all_genres"]:
            percent = round(probability * 100, 1)
            bar_color = "#6366f1" if percent >= 35 else "#334155"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin:5px 0'>"
                f"<span style='width:140px;color:#cbd5e1'>{genre.title()}</span>"
                f"<div style='flex:1;background:#1e293b;border-radius:4px;height:12px'>"
                f"<div style='width:{percent}%;background:{bar_color};height:100%;border-radius:4px'></div>"
                f"</div>"
                f"<span style='width:46px;text-align:right;color:#94a3b8;font-size:0.82rem'>{percent}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown("## 🎬 Movie Analyzer")
    st.markdown(
        "<p style='color:#94a3b8;margin-top:-10px'>Search any movie to get genre detection, sentiment analysis, quality verdict, predicted rating and similar films.</p>",
        unsafe_allow_html=True,
    )

    runtime = resolve_runtime()
    if runtime["mode"] == "none":
        st.error(runtime["error"])
        st.stop()

    mode_label = "Local model mode"
    st.markdown(f"<span class='mode-pill'>{mode_label}</span>", unsafe_allow_html=True)

    titles = runtime["titles"]
    title_lookup = {title.lower().strip(): title for title in titles}

    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    search_col, button_col = st.columns([5, 1])
    with search_col:
        query = st.text_input(
            "Search movie title",
            placeholder="e.g. Interstellar, The Room, Deadpool",
            label_visibility="collapsed",
        )
    with button_col:
        analyze = st.button("🔍 Analyze", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not query.strip():
        st.info("Type a movie title above to get started.", icon="🎥")
        return

    query_lower = query.strip().lower()
    exact_match = title_lookup.get(query_lower)
    suggestions = get_close_matches(query_lower, list(title_lookup.keys()), n=3, cutoff=0.45)

    canonical_title = exact_match or (title_lookup[suggestions[0]] if suggestions else query.strip())
    is_known_movie = canonical_title in titles

    if not exact_match and suggestions:
        selected_title = st.selectbox(
            "Did you mean…",
            [title_lookup[item] for item in suggestions] + ["Enter custom synopsis →"],
            index=0,
        )
        if selected_title == "Enter custom synopsis →":
            canonical_title = query.strip()
            is_known_movie = False
        else:
            canonical_title = selected_title
            is_known_movie = True

    synopsis_text = ""
    using_stored_text = False

    if is_known_movie:
        try:
            synopsis_text = get_known_movie_text(
                runtime["artifact"],
                canonical_title,
                base_dir=Path(".").resolve(),
                max_chars=40000,
            )
            using_stored_text = True
            st.caption("Using the stored local synopsis for this movie.")
        except ValueError:
            using_stored_text = False

    if not using_stored_text:
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.markdown("##### ✍️ Paste a synopsis or plot summary")
        synopsis_text = st.text_area(
            "Synopsis / Script excerpt",
            height=170,
            placeholder="Paste a brief synopsis or plot description here...",
            label_visibility="collapsed",
        ).strip()
        st.markdown("</div>", unsafe_allow_html=True)

    if not analyze:
        return

    if not using_stored_text and not synopsis_text:
        st.warning("Paste a synopsis above or choose a known movie title before analyzing.")
        return

    with st.spinner("Analyzing movie report…"):
        raw_result = local_prediction(
            runtime["artifact"],
            canonical_title,
            synopsis_text,
        )

    render_report(format_prediction_result(raw_result))


if __name__ == "__main__":
    main()
