from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from movie_analyzer import get_known_movie_text, load_model_artifact, run_prediction


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/movie_analyzer.h5"))
MODEL_BASE_DIR = Path(os.getenv("MODEL_BASE_DIR", ".")).resolve()

app = FastAPI(title="Movie Analyzer API", version="1.0.0")


class PredictRequest(BaseModel):
    movie_title: Optional[str] = None
    text: Optional[str] = None
    genre_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    top_k_genres: int = Field(default=6, ge=1, le=20)
    similar_k: int = Field(default=5, ge=1, le=20)


def extract_api_key(x_api_key: Optional[str], authorization: Optional[str]) -> str:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def verify_api_key(x_api_key: Optional[str], authorization: Optional[str]) -> None:
    expected_key = os.getenv("MODEL_API_KEY", "").strip()
    if not expected_key:
        return

    provided_key = extract_api_key(x_api_key, authorization)
    if provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@lru_cache(maxsize=1)
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run `python movie_analyzer.py train --base-dir .` first."
        )
    return load_model_artifact(MODEL_PATH)


@app.get("/health")
def health():
    artifact = load_artifact()
    return {
        "status": "ok",
        "movie_count": len(artifact["movies"]),
        "model_path": str(MODEL_PATH),
    }


@app.get("/titles")
def titles(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    verify_api_key(x_api_key, authorization)
    artifact = load_artifact()
    return {"movies": sorted(artifact["movies"])}


@app.post("/predict")
def predict(
    request: PredictRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    verify_api_key(x_api_key, authorization)
    artifact = load_artifact()

    if request.text:
        prediction_text = request.text
    elif request.movie_title:
        try:
            prediction_text = get_known_movie_text(
                artifact,
                request.movie_title,
                base_dir=MODEL_BASE_DIR,
                max_chars=40000,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=400, detail="Provide either `text` or a known `movie_title`.")

    return run_prediction(
        artifact=artifact,
        text=prediction_text,
        movie_title=request.movie_title,
        genre_threshold=request.genre_threshold,
        top_k_genres=request.top_k_genres,
        similar_k=request.similar_k,
    )
