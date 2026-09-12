from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import similarity_label


APP_PATH = Path(__file__).parents[1] / "app.py"


def test_similarity_labels_cover_score_ranges() -> None:
    assert similarity_label(0.90) == "Very similar"
    assert similarity_label(0.75) == "Moderately similar"
    assert similarity_label(0.60) == "Some shared structure"
    assert similarity_label(0.20) == "Structurally different"


def test_demo_interface_renders() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "🎵 MelodyMatch"
    assert app.radio[0].value == "Try the demo"
    assert len(app.selectbox) == 2
    assert any(metric.label == "Overall similarity" for metric in app.metric)


def test_upload_mode_has_two_uploaders() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.radio[0].set_value("Upload MIDI files").run()
    assert not app.exception
    assert len(app.get("file_uploader")) == 2
    assert any("Upload two MIDI files" in message.value for message in app.info)
