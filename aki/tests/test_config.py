from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from chandra_aca.transform import count_rate_to_mag

from aki import (
    BGD_MODE_DYNAMIC,
    BGD_MODE_IMAGE,
    BGD_MODE_LEGACY,
    BgdDynamic,
    BgdLegacy,
    BgdStatic,
    DarkStatic,
    SimConfig,
    SinusoidDither,
    simulate,
)
from aki.aki import (
    _reset_dyn_bgd_stack,
    calc_dynamic_bgd,
    calc_legacy_flight_bgd,
    get_pure_dither_samples,
    median5,
    star_track_numba,
)

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


def test_bgd_mode_and_image():
    bgd = BgdStatic.from_pixels({(0, 0): 5.0})
    assert bgd.mode == BGD_MODE_IMAGE
    assert bgd.image[512, 512] == 5.0
    legacy = BgdLegacy()
    assert legacy.mode == BGD_MODE_LEGACY
    assert legacy.image.shape == (0, 0)
    assert legacy.image.dtype == np.float64
    assert repr(legacy) == "BgdLegacy()"


def test_config_validation():
    with pytest.raises(ValueError):
        SimConfig(dark=DarkStatic.from_pixels({}), duration=0)
    with pytest.raises(ValueError):
        SinusoidDither(yaw_period=0)
    with pytest.raises(TypeError):
        SimConfig(DarkStatic.from_pixels({}))  # keyword-only


def test_scalar_fields_cast_to_float():
    dither = SinusoidDither(yaw_ampl=16, pitch_period=np.float64(500.0))
    assert type(dither.yaw_ampl) is float
    assert type(dither.pitch_period) is float
    assert "np.float64" not in repr(dither)
    cfg = SimConfig(dark=DarkStatic.from_pixels({}), duration=np.int64(100))
    assert type(cfg.duration) is float
    with pytest.raises(TypeError):
        SinusoidDither(yaw_ampl=np.array([1.0, 2.0]))


def test_star_track_numba_bgd_validation():
    dark = np.zeros((1024, 1024))
    with pytest.raises(TypeError, match="bgd must be"):
        star_track_numba(GUIDE, np.zeros(3), np.zeros(3), dark, STARS, bgd=dark)
    with pytest.raises(ValueError, match="same shape as dark"):
        star_track_numba(
            GUIDE,
            np.zeros(3),
            np.zeros(3),
            np.zeros((10, 10)),
            STARS,
            bgd=BgdStatic.from_pixels({}),
        )


def test_star_track_numba_bgd_static_matches_simulate():
    cfg = hot_pixel_config(bgd=BgdStatic.from_pixels({(0, 0): 1000.0}))
    out = simulate(GUIDE, STARS, cfg)
    exp = star_track_numba(
        GUIDE,
        *get_pure_dither_samples(DITHER, out["time"]),
        cfg.dark.get_image(),
        STARS,
        bgd=cfg.bgd,
    )
    for key in OUT_KEYS:
        np.testing.assert_array_equal(out[key], exp[key])


def test_median5():
    rng = np.random.default_rng(0)
    for vals in list(rng.uniform(size=(500, 5))) + list(rng.integers(0, 3, (500, 5))):
        assert median5(*vals) == np.median(vals)


def _new_dyn_stack(init):
    stack = np.empty((5, 16, 16))
    state = np.empty(3, dtype=np.int64)
    _reset_dyn_bgd_stack(stack, state, init)
    return stack, state


def test_calc_dynamic_bgd_median_of_stack():
    """Outer pixels need 3 of 5 samples before the median reflects them."""
    init = 30.0
    stack, state = _new_dyn_stack(init)
    img = np.full((8, 8), 100.0)
    img[0, 3] = 50.0  # outer pixel below init -> capped by image, not median
    args = (100, -40, stack, state, True, 500.0, False, init)

    bgd1 = calc_dynamic_bgd(img, *args)
    assert np.all(bgd1[img > init] == init)
    calc_dynamic_bgd(img, *args)
    bgd3 = calc_dynamic_bgd(img, *args)
    outer = np.ones((8, 8), dtype=bool)
    outer[1:7, 2:6] = False
    outer[2:6, 1:7] = False
    assert outer.sum() == 32
    np.testing.assert_array_equal(bgd3[outer], img[outer])
    np.testing.assert_array_equal(bgd3[~outer], init)
    assert state[2] == 3

    # Not tracking: stack is not updated
    calc_dynamic_bgd(img * 2, 100, -40, stack, state, False, 500.0, False, init)
    assert state[2] == 3

    # use_outer_min floors the inner pixels at the outer minimum (50)
    bgd = calc_dynamic_bgd(img, 100, -40, stack, state, True, 500.0, True, init)
    assert np.all(bgd[~outer] == 50.0)


def test_calc_dynamic_bgd_threshold_uses_legacy():
    stack, state = _new_dyn_stack(30.0)
    img = np.full((8, 8), 600.0)
    img[3, 3] = 5000.0
    bgd = calc_dynamic_bgd(img, 0, 0, stack, state, True, 500.0, False, 30.0)
    np.testing.assert_array_equal(bgd, calc_legacy_flight_bgd(img))
    # Stack not touched
    assert state[2] == 0
    assert np.all(stack == 30.0)


def test_calc_dynamic_bgd_stack_follows_window():
    """Stack samples stay attached to their CCD pixel as the window moves."""
    init = -1.0
    stack, state = _new_dyn_stack(init)
    ccd = np.random.default_rng(1).uniform(50, 100, (64, 64))
    rng = np.random.default_rng(2)
    r0, c0 = 28, 28
    for _ in range(300):
        r0 = int(np.clip(r0 + rng.integers(-2, 3), 0, 56))
        c0 = int(np.clip(c0 + rng.integers(-2, 3), 0, 56))
        img = ccd[r0 : r0 + 8, c0 : c0 + 8]
        calc_dynamic_bgd(img, r0, c0, stack, state, True, 500.0, False, init)
        # Window is inside the 16x16 stack
        assert 0 <= r0 - state[0] <= 8 and 0 <= c0 - state[1] <= 8
        # Every sampled stack value is the true CCD value at that absolute pixel
        rows, cols = np.ogrid[state[0] : state[0] + 16, state[1] : state[1] + 16]
        in_ccd = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
        truth = np.where(in_ccd, ccd[rows.clip(0, 63), cols.clip(0, 63)], np.nan)
        sampled = stack != init
        assert np.all(stack[sampled] == np.broadcast_to(truth, stack.shape)[sampled])
    assert np.count_nonzero(stack != init) > 200


def test_bgd_dynamic_config():
    bgd = BgdDynamic(outer_min_thresh=400, use_outer_min=1, stack_init=np.float64(5))
    assert bgd.mode == BGD_MODE_DYNAMIC
    assert bgd.image.shape == (0, 0)
    assert type(bgd.outer_min_thresh) is float
    assert bgd.use_outer_min is True
    assert type(bgd.stack_init) is float
    assert BgdDynamic().stack_init == pytest.approx(10 * 5 / 1.696)


def test_dynamic_bgd_reduces_hot_pixel_error():
    def rms_dy(bgd):
        out = simulate(GUIDE, STARS, hot_pixel_config(bgd=bgd))
        assert np.all(out["img_track"])
        dy = (out["cent_row"] - out["star_row"]) * 5
        return np.sqrt(np.mean(dy**2))

    cfg = hot_pixel_config()
    rms_legacy = rms_dy(BgdLegacy())
    rms_dynamic = rms_dy(BgdDynamic())
    rms_static = rms_dy(BgdStatic(image=cfg.dark.image))
    assert rms_static < rms_dynamic < rms_legacy / 2
