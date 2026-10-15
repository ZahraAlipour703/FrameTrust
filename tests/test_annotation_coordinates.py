import pytest

from scripts.annotate_video import normalize_display_click


def test_normalize_display_click_maps_preview_center():
    x, y = normalize_display_click(50, 25, 101, 51)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)


def test_normalize_display_click_clamps_to_preview_bounds():
    assert normalize_display_click(-5, 100, 101, 51) == (0.0, 1.0)


def test_normalize_display_click_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        normalize_display_click(0, 0, 0, 50)
