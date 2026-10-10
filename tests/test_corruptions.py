import numpy as np
import pytest

from frametrust.corruptions import corrupt_frame


def test_clean_is_unchanged_copy():
    image = np.full((100, 120, 3), 100, dtype=np.uint8)
    output = corrupt_frame(image, "clean", 0)
    assert np.array_equal(output, image)
    assert output is not image


def test_occlusion_changes_central_region():
    image = np.full((100, 120, 3), 200, dtype=np.uint8)
    output = corrupt_frame(image, "occlusion", 0)
    assert output[50, 50].tolist() == [0, 0, 0]
    assert output[0, 0].tolist() == [200, 200, 200]


def test_frame_drop_repeats_previous_frame():
    previous = np.full((20, 20, 3), 11, dtype=np.uint8)
    current = np.full((20, 20, 3), 99, dtype=np.uint8)
    output = corrupt_frame(current, "frame_drop", 5, previous_clean_frame=previous, drop_every=5)
    assert np.array_equal(output, previous)


def test_invalid_corruption_fails():
    with pytest.raises(ValueError):
        corrupt_frame(np.zeros((10, 10, 3), dtype=np.uint8), "noise", 0)
