# MelodyMatch: MIDI Melody Similarity Detector

MelodyMatch is a Python class project for symbolic MIDI analysis. It has two
separate parts:

- POP909 melody similarity: compares selected melodies with DTW-based pairwise
  similarity scoring.
- XMIDI genre classification: extracts explainable symbolic MIDI features and
  trains a decision tree to predict genre from XMIDI filenames.

The project works with MIDI note events, not audio waveforms.

## Project Structure

```text
melodymatch/
  __init__.py
  config.py
  features.py
  midi_loader.py
  models.py
  pipeline.py
  similarity.py
  xmidi/
    __init__.py
    dataset.py             # Scan XMIDI_Dataset and list MIDI files
    feature_extraction.py  # Symbolic MIDI features for genre classification
    genre_model.py         # Decision tree training, evaluation, model save/load
    labels.py              # Parse XMIDI_<Emotion>_<Genre>_<ID>.midi filenames
    pipeline.py            # High-level XMIDI train/predict workflows
app.py                     # Streamlit UI for both project parts
run_backend.py             # POP909 similarity command-line runner
run_genre_classifier.py    # XMIDI genre-classifier command-line runner
requirements.txt
README.md
```

## POP909 Melody Similarity

For each selected POP909 song, MelodyMatch expects this layout:

```text
POP909/
  001/
    001.mid
    beat_midi.txt
    beat_audio.txt
```

By default, the selected songs are:

```text
001 002 003 004 005
```

The POP909 loader uses:

- `<song_id>.mid`
- `beat_midi.txt` when available
- `beat_audio.txt` when `beat_midi.txt` is not available

It ignores:

- `chord_audio.txt`
- `chord_midi.txt`
- `key_audio.txt`
- `versions/`

The main melody-track rule is to extract the MIDI track whose name contains
`MELODY`, case-insensitively. If no track is explicitly named `MELODY`, the
fallback selects the non-drum track with the highest average pitch, using note
count as a tie-breaker.

For each melody, the system extracts:

- pitch sequence
- interval sequence
- note durations
- contour sequence
- beat placement

Pairwise similarity uses dynamic time warping over interval, duration, contour,
and beat-placement features. The final score is a weighted average:

```python
DEFAULT_SIMILARITY_WEIGHTS = {
    "interval": 0.40,
    "duration": 0.30,
    "contour": 0.20,
    "beat": 0.10,
}
```

The POP909 backend writes:

- `outputs/pairwise_similarity.csv`

Run it with:

```bash
python run_backend.py
```

Or specify the dataset path:

```bash
python run_backend.py --dataset-root /path/to/POP909 --songs 001 002 003 004 005
```

## XMIDI Genre Classification Extension

The XMIDI extension uses the local folder:

```text
XMIDI_Dataset/
```

Each MIDI filename encodes its labels:

```text
XMIDI_<Emotion>_<Genre>_<ID>.midi
```

For example:

```text
XMIDI_happy_pop_SN74XHLX.midi
```

is parsed as:

- emotion: `happy`
- genre: `pop`
- file id: `SN74XHLX`

The genre label is parsed from the filename. No separate metadata CSV is used.

### XMIDI Features

Each XMIDI row contains the filename labels plus symbolic MIDI features such as:

- note count
- non-drum note count
- drum-note ratio
- total duration
- note density
- average note duration
- duration variance
- average pitch
- pitch variance
- pitch range
- average velocity
- velocity variance
- average and max polyphony estimates
- overlap ratio
- estimated tempo
- tempo-change count
- mean inter-onset interval
- inter-onset interval variance
- rhythmic variability
- interval mean and standard deviation
- average absolute interval
- contour up/down/same ratios
- absolute interval histogram ratios

Messy or invalid MIDI files are skipped and recorded in:

- `outputs/xmidi_feature_errors.csv`

### XMIDI Model

The main model is:

- `sklearn.tree.DecisionTreeClassifier`

This keeps the approach explainable and lightweight. The training script also
reports a small random forest baseline accuracy by default, but the saved model
and decision-tree rules come from the decision tree.

The XMIDI training pipeline:

1. Scans `XMIDI_Dataset/`
2. Parses emotion, genre, and file id from filenames
3. Extracts symbolic MIDI features
4. Builds a feature table
5. Splits into train and test sets
6. Trains the decision tree to predict genre
7. Prints accuracy, classification report, and confusion matrix
8. Saves CSV outputs and a model artifact

XMIDI outputs:

- `outputs/xmidi_features.csv`
- `outputs/xmidi_test_predictions.csv`
- `outputs/xmidi_confusion_matrix.csv`
- `outputs/xmidi_feature_errors.csv`
- `outputs/xmidi_genre_decision_tree.joblib`

Train on the full local XMIDI dataset:

```bash
python run_genre_classifier.py
```

Run a smaller smoke test:

```bash
python run_genre_classifier.py --max-files 1000
```

Reuse an existing feature CSV:

```bash
python run_genre_classifier.py --reuse-features
```

## Installation

Create and activate a virtual environment, then install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Streamlit UI

Start the UI:

```bash
streamlit run app.py
```

The sidebar has two pages:

- `POP909 Melody Similarity`
- `XMIDI Genre Classification`

The POP909 page lets you choose two songs and shows:

- final similarity score
- interval score
- duration score
- contour score
- beat score
- all pairwise similarity results

The XMIDI page lets you:

- scan the local `XMIDI_Dataset/`
- train the decision tree model
- choose one XMIDI file
- predict its genre
- see the parsed emotion label
- see model confidence when available
- inspect core extracted MIDI features

