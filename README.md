# MelodyMatch

An interactive music-information-retrieval project for comparing symbolic
melodies and explaining what makes them sound similar.

MelodyMatch extracts the lead melody from MIDI files, aligns musical sequences
with dynamic time warping, and reports an overall similarity score alongside
four interpretable components:

- pitch intervals
- note duration
- melodic contour
- beat placement

The repository includes five small POP909 examples, so the interface works
immediately after installation. It analyzes MIDI note events rather than audio
recordings.

## How it works

Each MIDI file is reduced to a melody feature sequence. Dynamic time warping
aligns sequences of different lengths before each component receives a score
from 0 to 1. The final score uses transparent, configurable weights:

| Component | Weight | What it captures |
|---|---:|---|
| Pitch intervals | 40% | Movement between consecutive notes |
| Note duration | 30% | Relative rhythmic lengths |
| Melodic contour | 20% | Upward, downward, or repeated motion |
| Beat placement | 10% | Position within the beat cycle |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To run the command-line analysis and export all pairwise results:

```bash
python run_backend.py
```

The CSV is written to `outputs/pairwise_similarity.csv`.

## Project structure

```text
MelodyMatch/
├── app.py                  # Streamlit comparison interface
├── run_backend.py          # Command-line analysis
├── melodymatch/
│   ├── features.py         # Musical feature extraction
│   ├── midi_loader.py      # MIDI and beat-data loading
│   ├── pipeline.py         # Reusable analysis workflow
│   └── similarity.py       # DTW and weighted scoring
├── data/POP909/            # Five demonstration melodies
└── requirements.txt
```

## Data attribution

The bundled demonstration files are excerpts from the
[POP909 dataset](https://github.com/music-x-lab/POP909-Dataset), distributed
under its MIT License. See [`data/POP909/NOTICE.md`](data/POP909/NOTICE.md) for
the source and academic citation.

## Current scope

This version is intentionally focused on explainable melody comparison. The
next iteration will add richer visual explanations, MIDI upload, tests, and a
deployed demo.

## Author

**Selorm Essey**

M.S. Computer Science, Emory University
