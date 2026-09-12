from melodymatch.features import beat_phase, contour_from_interval


def test_contour_maps_interval_direction() -> None:
    assert contour_from_interval(-4) == -1
    assert contour_from_interval(0) == 0
    assert contour_from_interval(7) == 1


def test_beat_phase_reports_position_between_beats() -> None:
    assert beat_phase(1.5, [0.0, 1.0, 2.0]) == 0.5
    assert beat_phase(1.0, [0.0, 1.0, 2.0]) == 0.0


def test_beat_phase_requires_two_beat_markers() -> None:
    assert beat_phase(0.5, []) is None
    assert beat_phase(0.5, [0.0]) is None
