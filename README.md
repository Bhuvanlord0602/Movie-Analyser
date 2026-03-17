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
- Saved model: `models/movie_analyzer.joblib`
- Cleaned training rows: `models/prepared_training_data.csv`

## 3. Run the Streamlit app locally

```powershell
streamlit run streamlit_app.py
```

If `models/movie_analyzer.joblib` is missing, the app will attempt to train it automatically on first run.

## 4. Predict using a synopsis file

```powershell
python movie_analyzer.py predict --base-dir "." --movie-title "Interstellar" --text-file "Movie scripts/scirpt/interstellar.txt"
```

## 5. Predict using raw text

```powershell
python movie_analyzer.py predict --base-dir "." --movie-title "My New Movie" --text "A team of astronauts travels through a wormhole to save humanity."
```

## 6. Predict from an existing movie title in trained data

If the title exists and has linked script text:

```powershell
python movie_analyzer.py predict --base-dir "." --movie-title "The Room"
```

## Notes

- Your CSV has some filename/title mismatches and duplicates. The trainer automatically cleans and matches records.
- If a synopsis file is missing, the row is still used with fallback text so training does not fail.
- Training metrics printed by the script are fit metrics on available data, useful for quick validation.
