from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from chandra_aca.transform import count_rate_to_mag

from aki import (
    BGD_MODE_IMAGE,
    BgdLegacy,
    BgdStatic,
    DarkStatic,
    SimConfig,
    SinusoidDither,
    simulate,
)
from aki.aki import get_pure_dither_samples, star_track_numba

STAR_NORM = 10000.0
GUIDE = {"row": 0.0, "col": 0.0, "maxmag": 11.2, "slot": 0}
STARS = {
    "row": np.array([0.0]),
    "col": np.array([0.0]),
    "mag": np.array([count_rate_to_mag(STAR_NORM)]),
}
DITHER = SinusoidDither(
    yaw_ampl=16.0, pitch_ampl=16.0, yaw_period=707.0, pitch_period=1000.0
)
OUT_KEYS = ["star_row", "star_col", "cent_row", "cent_col", "img_sum", "img_track"]


def hot_pixel_config(**kwargs):
    return SimConfig(
        duration=5000.0,
        dither=DITHER,
        dark=DarkStatic.from_pixels({(0, 0): 1000.0}),
        **kwargs,
    )


def test_simulate_legacy_matches_star_track_numba():
    """simulate() with BgdLegacy reproduces calling the kernel directly."""
    cfg = hot_pixel_config()
    out = simulate(GUIDE, STARS, cfg)

    times = np.arange(0.0, 5000.0, 2.05)
    dither_rs, dither_cs = get_pure_dither_samples(DITHER, times)
    dark = np.zeros((1024, 1024))
    dark[512, 512] = 1000.0
    exp = star_track_numba(GUIDE, dither_rs, dither_cs, dark, STARS)

    assert out["config"] is cfg
    assert len(out["time"]) == 2440
    for key in OUT_KEYS + ["time"]:
        np.testing.assert_array_equal(out[key], exp[key])

    # Hot pixel produces arcsec-level centroid errors with the legacy background
    dy = (out["cent_row"] - out["star_row"]) * 5
    assert 1.5 < np.max(np.abs(dy)) < 2.5


def test_image_bgd_removes_known_hot_pixel():
    """Subtracting the exact dark current makes the hot pixel invisible."""
    cfg_hot = hot_pixel_config()
    cfg_hot = replace(cfg_hot, bgd=BgdStatic(image=cfg_hot.dark.image))
    cfg_zero = replace(
        cfg_hot,
        dark=DarkStatic.from_pixels({}),
        bgd=BgdStatic.from_pixels({}),
    )
    out_hot = simulate(GUIDE, STARS, cfg_hot)
    out_zero = simulate(GUIDE, STARS, cfg_zero)
    for key in OUT_KEYS:
        np.testing.assert_allclose(out_hot[key], out_zero[key], rtol=0, atol=1e-9)

    # Residuals are then just the small PSF-grid / centroid baseline
    dy = (out_hot["cent_row"] - out_hot["star_row"]) * 5
    assert np.max(np.abs(dy)) < 0.5


def test_image_bgd_too_low_offsets_centroid():
    """A background 1000 e-/s too low for the hot pixel offsets the centroid."""
    cfg = hot_pixel_config(bgd=BgdStatic.from_pixels({}))
    out = simulate(GUIDE, STARS, cfg)
    dy = (out["cent_row"] - out["star_row"]) * 5
    assert np.max(np.abs(dy)) > 1.0


def test_static_dark_from_pixels_and_readonly():
    dark = DarkStatic.from_pixels({(0, 0): 1000.0, (-3, 5): 50.0}, background=2.0)
    assert dark.image[512, 512] == 1000.0
    assert dark.image[509, 517] == 50.0
    assert dark.image[0, 0] == 2.0
    with pytest.raises(ValueError):
        dark.image[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        dark.image = np.zeros((1024, 1024))
    assert "n_nonzero" in repr(dark)


def test_static_dark_copies_input():
    image = np.zeros((1024, 1024))
    dark = DarkStatic(image=image)
    image[0, 0] = 99.0
    assert dark.image[0, 0] == 0.0


@pytest.mark.parametrize("cls", [DarkStatic, BgdStatic])
def test_bad_image_shape(cls):
    with pytest.raises(ValueError, match="shape"):
        cls(image=np.zeros((8, 8)))


def test_bgd_prepare():
    mode, img = BgdStatic.from_pixels({(0, 0): 5.0}).prepare()
    assert mode == BGD_MODE_IMAGE
    assert img[512, 512] == 5.0
    assert BgdLegacy().prepare()[1].shape == (0, 0)


def test_config_validation():
    with pytest.raises(ValueError):
        SimConfig(dark=DarkStatic.from_pixels({}), duration=0)
    with pytest.raises(ValueError):
        SinusoidDither(yaw_period=0)
    with pytest.raises(TypeError):
        SimConfig(DarkStatic.from_pixels({}))  # keyword-only


def test_star_track_numba_image_mode_requires_bgd_img():
    dark = np.zeros((1024, 1024))
    with pytest.raises(ValueError, match="bgd_img"):
        star_track_numba(
            GUIDE, np.zeros(3), np.zeros(3), dark, STARS, bgd_mode=BGD_MODE_IMAGE
        )
