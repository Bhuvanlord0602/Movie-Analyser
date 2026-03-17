from __future__ import annotations

import argparse
import importlib
import json
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer


def normalize_key(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_genre(genre: str) -> str:
    value = str(genre or "").strip().lower()
    value = value.replace("sci fi", "sci-fi").replace("scifi", "sci-fi")
    value = re.sub(r"\s+", " ", value)
    return value


def parse_genres(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    pieces = [normalize_genre(part) for part in str(value).split(",")]
    return [item for item in pieces if item]


def choose_scripts_dir(base_dir: Path, user_value: Optional[str]) -> Path:
    if user_value:
        resolved = Path(user_value)
        if not resolved.is_absolute():
            resolved = (base_dir / resolved).resolve()
        return resolved

    candidates = [
        base_dir / "Movie scripts" / "scirpt",
        base_dir / "Movie scripts" / "script",
        base_dir / "Movie scripts",
        base_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find a scripts folder. Pass --scripts-dir explicitly.")


@dataclass
class ScriptLookup:
    by_key: Dict[str, Path]
    keys: List[str]


class ScriptResolver:
    def __init__(self, scripts_dir: Path):
        self.scripts_dir = scripts_dir
        self.lookup = self._build_lookup()

    def _build_lookup(self) -> ScriptLookup:
        by_key: Dict[str, Path] = {}
        for file_path in self.scripts_dir.rglob("*.txt"):
            if file_path.name.lower() == "desktop.ini":
                continue
            candidates = {
                normalize_key(file_path.name),
                normalize_key(file_path.stem),
            }
            for key in candidates:
                if key and key not in by_key:
                    by_key[key] = file_path
        return ScriptLookup(by_key=by_key, keys=list(by_key.keys()))

    def find(self, filename: str, movie_title: str) -> Tuple[Optional[Path], float]:
        raw_candidates = [filename, Path(filename).stem if filename else "", movie_title]
        candidates = [normalize_key(item) for item in raw_candidates if normalize_key(item)]

        for candidate in candidates:
            if candidate in self.lookup.by_key:
                return self.lookup.by_key[candidate], 1.0

        best_path: Optional[Path] = None
        best_score = 0.0
        for candidate in candidates:
            for key in self.lookup.keys:
                score = SequenceMatcher(a=candidate, b=key).ratio()
                if score > best_score:
                    best_score = score
                    best_path = self.lookup.by_key[key]

        if best_path is not None and best_score >= 0.86:
            return best_path, best_score
        return None, 0.0


def assign_sentiment(rating: float) -> str:
    if rating >= 7.0:
        return "positive"
    if rating <= 5.5:
        return "negative"
    return "neutral"


def read_text(file_path: Path, max_chars: int = 25000) -> str:
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    content = re.sub(r"\s+", " ", content).strip()
    return content[:max_chars]


def relativize_path(path_value: str, base_dir: Path) -> str:
    if not path_value:
        return ""

    path = Path(path_value)
    if not path.is_absolute():
        return str(path)

    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def resolve_saved_path(path_value: str, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_value)
    if path.is_absolute() or base_dir is None:
        return path
    return (base_dir / path).resolve()


def get_known_movie_text(
    artifact: Dict[str, Any],
    movie_title: str,
    base_dir: Optional[Path] = None,
    max_chars: int = 40000,
) -> str:
    wanted = normalize_key(movie_title)
    for title, path in zip(artifact["movies"], artifact["script_paths"]):
        if normalize_key(title) == wanted and path:
            return read_text(resolve_saved_path(path, base_dir), max_chars=max_chars)

    raise ValueError(f"No linked synopsis text found for movie title: {movie_title}")


def import_h5py_module():
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for .h5 artifacts. Install it with `pip install h5py`.") from exc


def save_model_artifact(artifact: Dict[str, Any], model_path: Path) -> None:
    suffix = model_path.suffix.lower()

    if suffix == ".h5":
        h5py = import_h5py_module()

        payload = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
        data = np.frombuffer(payload, dtype=np.uint8)
        with h5py.File(model_path, "w") as handle:
            handle.attrs["artifact_format"] = "movie_analyzer_pickle_v1"
            handle.create_dataset("payload", data=data, compression="gzip", compression_opts=9)
        return

    if suffix == ".pt":
        with model_path.open("wb") as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    joblib.dump(artifact, model_path)


def load_model_artifact(model_path: Path) -> Dict[str, Any]:
    suffix = model_path.suffix.lower()

    if suffix == ".h5":
        h5py = import_h5py_module()

        with h5py.File(model_path, "r") as handle:
            payload = handle["payload"][()].tobytes()
        return pickle.loads(payload)

    if suffix == ".pt":
        with model_path.open("rb") as handle:
            return pickle.load(handle)

    return joblib.load(model_path)


def merge_unique_lists(values: Iterable[List[str]]) -> List[str]:
    output: List[str] = []
    seen = set()
    for items in values:
        for item in items:
            if item not in seen:
                seen.add(item)
                output.append(item)
    return output


def load_and_prepare_data(csv_path: Path, scripts_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [column.strip().lower() for column in df.columns]

    required = {"movie", "ratings", "filename", "genre"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df["movie"] = df["movie"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["filename"] = df["filename"].fillna("").astype(str).str.strip()
    df["ratings"] = pd.to_numeric(df["ratings"], errors="coerce")
    df["genres"] = df["genre"].apply(parse_genres)

    df = df[df["ratings"].notna() & df["movie"].ne("")].copy()
    df["record_key"] = np.where(
        df["filename"].str.strip().ne(""),
        df["filename"],
        df["movie"],
    )
    df["record_key"] = df["record_key"].apply(normalize_key)

    df = (
        df.sort_values("ratings", ascending=False)
        .groupby("record_key", as_index=False)
        .agg(
            movie=("movie", "first"),
            filename=("filename", "first"),
            ratings=("ratings", "mean"),
            genres=("genres", merge_unique_lists),
        )
    )

    resolver = ScriptResolver(scripts_dir)

    matched_paths: List[Optional[Path]] = []
    matched_scores: List[float] = []
    script_texts: List[str] = []

    for _, row in df.iterrows():
        path, score = resolver.find(row["filename"], row["movie"])
        matched_paths.append(path)
        matched_scores.append(score)

        if path is not None:
            script_texts.append(read_text(path))
        else:
            # Keep training robust even if a synopsis file is missing.
            script_texts.append("")

    df["script_path"] = [str(path) if path else "" for path in matched_paths]
    df["match_score"] = matched_scores
    df["script_found"] = df["script_path"].ne("")
    df["script_text"] = script_texts
    df["model_text"] = (
        df["movie"].str.strip()
        + " . "
        + np.where(df["script_text"].str.strip().ne(""), df["script_text"], "No synopsis available.")
    )

    df["sentiment"] = df["ratings"].apply(assign_sentiment)
    df["good_bad"] = np.where(df["ratings"] >= 6.5, "good", "bad")

    return df


def ensure_at_least_one_genre(binary_predictions: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    fixed = binary_predictions.copy()
    for index in range(fixed.shape[0]):
        if fixed[index].sum() == 0:
            best_class = int(np.argmax(probabilities[index]))
            fixed[index, best_class] = 1
    return fixed


def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.92,
        min_df=1,
        max_features=40000,
        sublinear_tf=True,
    )
    x_all = vectorizer.fit_transform(df["model_text"])

    mlb = MultiLabelBinarizer()
    y_genre = mlb.fit_transform(df["genres"])

    genre_model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
        )
    )
    genre_model.fit(x_all, y_genre)

    sentiment_model = LogisticRegression(max_iter=3000, class_weight="balanced")
    sentiment_model.fit(x_all, df["sentiment"])

    good_bad_model = LogisticRegression(max_iter=3000, class_weight="balanced")
    good_bad_model.fit(x_all, df["good_bad"])

    rating_model = Ridge(alpha=1.0)
    rating_model.fit(x_all, df["ratings"])

    neighbors = NearestNeighbors(metric="cosine", n_neighbors=min(8, len(df)))
    neighbors.fit(x_all)

    genre_prob = genre_model.predict_proba(x_all)
    genre_pred = ensure_at_least_one_genre((genre_prob >= 0.35).astype(int), genre_prob)

    sentiment_pred = sentiment_model.predict(x_all)
    good_bad_pred = good_bad_model.predict(x_all)
    rating_pred = rating_model.predict(x_all)

    metrics = {
        "genre_micro_f1_train": float(f1_score(y_genre, genre_pred, average="micro", zero_division=0)),
        "sentiment_accuracy_train": float(accuracy_score(df["sentiment"], sentiment_pred)),
        "good_bad_accuracy_train": float(accuracy_score(df["good_bad"], good_bad_pred)),
        "rating_mae_train": float(mean_absolute_error(df["ratings"], rating_pred)),
    }

    return {
        "vectorizer": vectorizer,
        "genre_model": genre_model,
        "mlb": mlb,
        "sentiment_model": sentiment_model,
        "good_bad_model": good_bad_model,
        "rating_model": rating_model,
        "neighbors": neighbors,
        "x_all": x_all,
        "movies": df["movie"].tolist(),
        "ratings": df["ratings"].astype(float).tolist(),
        "genres": df["genres"].tolist(),
        "sentiments": df["sentiment"].tolist(),
        "good_bad_labels": df["good_bad"].tolist(),
        "script_paths": df["script_path"].tolist(),
        "metrics": metrics,
    }


def top_genres(mlb: MultiLabelBinarizer, probabilities: np.ndarray, threshold: float, top_k: int) -> List[Dict[str, Any]]:
    class_probs = list(zip(mlb.classes_, probabilities.tolist()))
    class_probs.sort(key=lambda item: item[1], reverse=True)

    selected = [
        {"genre": genre, "probability": round(float(prob), 4)}
        for genre, prob in class_probs
        if prob >= threshold
    ]

    if not selected and class_probs:
        best_genre, best_prob = class_probs[0]
        selected = [{"genre": best_genre, "probability": round(float(best_prob), 4)}]

    return selected[:top_k]


def similar_movies(
    artifact: Dict[str, Any],
    input_vector,
    movie_title: Optional[str],
    count: int,
) -> List[Dict[str, Any]]:
    count = max(1, count)
    limit = min(count + 1, len(artifact["movies"]))
    distances, indices = artifact["neighbors"].kneighbors(input_vector, n_neighbors=limit)

    normalized_input_title = normalize_key(movie_title or "")
    output: List[Dict[str, Any]] = []

    for distance, idx in zip(distances[0], indices[0]):
        candidate_title = artifact["movies"][idx]
        if normalized_input_title and normalize_key(candidate_title) == normalized_input_title:
            continue
        output.append(
            {
                "movie": candidate_title,
                "similarity": round(float(1.0 - distance), 4),
                "known_rating": round(float(artifact["ratings"][idx]), 2),
                "known_genres": artifact["genres"][idx],
            }
        )
        if len(output) >= count:
            break

    return output


def run_prediction(
    artifact: Dict[str, Any],
    text: str,
    movie_title: Optional[str],
    genre_threshold: float,
    top_k_genres: int,
    similar_k: int,
) -> Dict[str, Any]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        raise ValueError("Prediction text is empty. Provide --text, --text-file, or a known --movie-title.")

    model_text = f"{(movie_title or 'Unknown Movie').strip()} . {cleaned}"
    vector = artifact["vectorizer"].transform([model_text])

    genre_probabilities = artifact["genre_model"].predict_proba(vector)[0]
    sentiment_prob = artifact["sentiment_model"].predict_proba(vector)[0]
    good_bad_prob = artifact["good_bad_model"].predict_proba(vector)[0]

    sentiment_classes = artifact["sentiment_model"].classes_
    good_bad_classes = artifact["good_bad_model"].classes_

    all_genre_probabilities = sorted(
        [
            {"genre": genre, "probability": round(float(probability), 4)}
            for genre, probability in zip(artifact["mlb"].classes_, genre_probabilities.tolist())
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )

    sentiment_index = int(np.argmax(sentiment_prob))
    good_index = list(good_bad_classes).index("good") if "good" in good_bad_classes else int(np.argmax(good_bad_prob))

    predicted_rating = float(np.clip(artifact["rating_model"].predict(vector)[0], 0.0, 10.0))

    return {
        "movie": movie_title or "Unknown Movie",
        "predicted_genres": top_genres(artifact["mlb"], genre_probabilities, genre_threshold, top_k_genres),
        "all_genre_probabilities": all_genre_probabilities,
        "predicted_sentiment": {
            "label": str(sentiment_classes[sentiment_index]),
            "confidence": round(float(sentiment_prob[sentiment_index]), 4),
            "class_probabilities": {
                str(label): round(float(prob), 4) for label, prob in zip(sentiment_classes, sentiment_prob)
            },
        },
        "good_or_bad": {
            "label": "good" if good_bad_prob[good_index] >= 0.5 else "bad",
            "probability_good": round(float(good_bad_prob[good_index]), 4),
        },
        "predicted_rating": round(predicted_rating, 2),
        "similar_movies": similar_movies(artifact, vector, movie_title, similar_k),
    }


def train_command(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).resolve()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (base_dir / csv_path).resolve()

    scripts_dir = choose_scripts_dir(base_dir, args.scripts_dir)
    model_path = Path(args.model_out)
    if not model_path.is_absolute():
        model_path = (base_dir / model_path).resolve()

    prepared_path = Path(args.prepared_csv)
    if not prepared_path.is_absolute():
        prepared_path = (base_dir / prepared_path).resolve()

    df = load_and_prepare_data(csv_path, scripts_dir)
    df["script_path"] = df["script_path"].apply(lambda value: relativize_path(value, base_dir))
    artifact = train_models(df)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)

    save_model_artifact(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "csv_path": str(csv_path),
            "scripts_dir": str(scripts_dir),
            **artifact,
        },
        model_path,
    )

    export_df = df[[
        "movie",
        "ratings",
        "filename",
        "genres",
        "sentiment",
        "good_bad",
        "script_found",
        "script_path",
        "match_score",
    ]].copy()
    export_df["genres"] = export_df["genres"].apply(lambda values: ", ".join(values))
    export_df.to_csv(prepared_path, index=False)

    summary = {
        "rows_used": int(len(df)),
        "scripts_found": int(df["script_found"].sum()),
        "scripts_missing": int((~df["script_found"]).sum()),
        "unique_genres": int(len(artifact["mlb"].classes_)),
        "metrics_train": artifact["metrics"],
        "model_saved_to": str(model_path),
        "prepared_data_saved_to": str(prepared_path),
    }
    print(json.dumps(summary, indent=2))


def read_prediction_text(args: argparse.Namespace, artifact: Dict[str, Any]) -> str:
    if args.text:
        return args.text

    if args.text_file:
        text_file = Path(args.text_file)
        if not text_file.is_absolute():
            text_file = (Path(args.base_dir).resolve() / text_file).resolve()
        return read_text(text_file, max_chars=40000)

    if args.movie_title:
        return get_known_movie_text(
            artifact,
            args.movie_title,
            base_dir=Path(args.base_dir).resolve(),
            max_chars=40000,
        )

    raise ValueError("Provide --text, --text-file, or a known --movie-title with available synopsis text.")


def predict_command(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).resolve()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (base_dir / model_path).resolve()

    artifact = load_model_artifact(model_path)
    text = read_prediction_text(args, artifact)

    output = run_prediction(
        artifact=artifact,
        text=text,
        movie_title=args.movie_title,
        genre_threshold=args.genre_threshold,
        top_k_genres=args.top_k_genres,
        similar_k=args.similar_k,
    )

    print(json.dumps(output, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a movie analyzer that predicts genres, sentiment, good/bad label, and rating.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model artifacts from CSV + synopsis files.")
    train_parser.add_argument(
        "--base-dir",
        default=".",
        help="Workspace base directory used to resolve relative paths.",
    )
    train_parser.add_argument("--csv", default="Movie data - Sheet1.csv", help="Path to training CSV file.")
    train_parser.add_argument(
        "--scripts-dir",
        default=None,
        help="Folder that contains synopsis/script .txt files. If omitted, auto-detected.",
    )
    train_parser.add_argument(
        "--model-out",
        default="models/movie_analyzer.h5",
        help="Path to save trained artifact.",
    )
    train_parser.add_argument(
        "--prepared-csv",
        default="models/prepared_training_data.csv",
        help="Path to save cleaned/linked training rows for inspection.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict movie properties from new text.")
    predict_parser.add_argument(
        "--base-dir",
        default=".",
        help="Workspace base directory used to resolve relative paths.",
    )
    predict_parser.add_argument(
        "--model",
        default="models/movie_analyzer.h5",
        help="Path to trained artifact.",
    )
    predict_parser.add_argument("--movie-title", default=None, help="Movie title for reporting/context.")
    predict_parser.add_argument("--text", default=None, help="Raw synopsis/script text for prediction.")
    predict_parser.add_argument("--text-file", default=None, help="Path to a .txt file with synopsis/script text.")
    predict_parser.add_argument(
        "--genre-threshold",
        type=float,
        default=0.35,
        help="Probability threshold for genre selection.",
    )
    predict_parser.add_argument(
        "--top-k-genres",
        type=int,
        default=5,
        help="Maximum number of predicted genres to return.",
    )
    predict_parser.add_argument(
        "--similar-k",
        type=int,
        default=5,
        help="How many similar movies to include.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
