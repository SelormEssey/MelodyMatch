from melodymatch.pipeline import load_selected_features, parse_song_ids
from melodymatch.similarity import compare_all_pairs


def test_song_ids_are_normalized() -> None:
    assert parse_song_ids("1, 02 003") == ["001", "002", "003"]


def test_five_demo_songs_create_ten_unique_pairs() -> None:
    features = load_selected_features("data/POP909")
    pairwise = compare_all_pairs(features)
    assert len(features) == 5
    assert len(pairwise) == 10
    assert pairwise["final_score"].between(0, 1).all()
