# Movie Analyzer (Genre + Sentiment + Good/Bad + Rating)

This project trains a machine learning model from your movie CSV and synopsis/script text files.

The public Streamlit entrypoint is `streamlit_app.py`.

It predicts:
- Genre (multi-label)
- Sentiment (`positive`, `neutral`, `negative`)
- Good/Bad label
- Predicted rating (0 to 10)
- Similar movies from your dataset

## 1. Install dependencies

```powershell
pip install -r requirements.txt
```

## 2. Train the model

From this workspace root folder:

```powershell
python movie_analyzer.py train --base-dir "."
```

Optional explicit scripts directory:

```powershell
python movie_analyzer.py train --base-dir "." --scripts-dir "Movie scripts/scirpt"
```

Training output:
- Saved model (default): `models/movie_analyzer.h5`
- Cleaned training rows: `models/prepared_training_data.csv`

Optional model format:

```powershell
python movie_analyzer.py train --base-dir "." --model-out "models/movie_analyzer.pt"
```

`streamlit_app.py` and `api_server.py` automatically load in this priority:
1. `models/movie_analyzer.h5`
2. `models/movie_analyzer.pt`
3. `models/movie_analyzer.joblib`

## 3. Run the Streamlit app locally

```powershell
streamlit run streamlit_app.py
```

## 4. Run the protected model API locally

Set an API key in your shell first:

```powershell
$env:MODEL_API_KEY = "replace-with-your-secret-key"
```

Then start the backend:

```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 5. Connect Streamlit to the protected API

Create `.streamlit/secrets.toml` locally using `.streamlit/secrets.toml.example` as a template:

```toml
MODEL_API_URL = "http://127.0.0.1:8000"
MODEL_API_KEY = "replace-with-your-secret-key"
```

When these secrets are present, `streamlit_app.py` calls the protected backend API. When they are not present, the app falls back to the local model artifact.

## 6. Predict using a synopsis file

```powershell
python movie_analyzer.py predict --base-dir "." --movie-title "Interstellar" --text-file "Movie scripts/scirpt/interstellar.txt"
```

## 7. Predict using raw text

```powershell
python movie_analyzer.py predict --base-dir "." --movie-title "My New Movie" --text "A team of astronauts travels through a wormhole to save humanity."
```

## 8. Predict from an existing movie title in trained data

If the title exists and has linked script text:

```powershell
python movie_analyzer.py predict --base-dir "." --movie-title "The Room"
```

## Notes

- Your CSV has some filename/title mismatches and duplicates. The trainer automatically cleans and matches records.
- If a synopsis file is missing, the row is still used with fallback text so training does not fail.
- Training metrics printed by the script are fit metrics on available data, useful for quick validation.
- Do not commit `.streamlit/secrets.toml`. Keep API keys only in Streamlit secrets or environment variables.
