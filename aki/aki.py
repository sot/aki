import itertools

import agasc
import astropy.table as apt
import numba
import numpy as np
from chandra_aca import transform
from chandra_aca.aca_image import AcaPsfLibrary
from mica.archive.aca_dark import get_dark_cal_image

__all__ = ["calc_legacy_flight_bgd", "centroid_fm"]

# Row and col positions for legacy flight background for calc_legacy_flight_bgd.
# Performance note: keeping this outside the function is faster.
ROW_BGD = np.array([0, 0, 0, 0, 7, 7, 7, 7])
COL_BGD = np.array([0, 1, 6, 7, 0, 1, 6, 7])
CLIP_MIN_BGD = 10 * 5 / 1.696  # 10 DN in e-/s for 1.696 second integration

MOUSE_BIT_COLS = np.concatenate(
    [
        [2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5],
    ]
)
MOUSE_BIT_ROWS = np.concatenate(
    [
        [1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
        [6, 6, 6, 6],
    ]
)

APL = AcaPsfLibrary()


def make_psf_images(n_rc=11):
    """Grid of PSF images for rows/cols in range -0.5 to 0.5 with n_rc points."""
    rc_vals = np.linspace(-0.5, 0.5, n_rc)
    cols, rows = np.meshgrid(rc_vals, rc_vals)
    imgs = np.zeros(rows.shape + (8, 8), dtype=np.float64)
    for ii in range(rows.shape[0]):
        for jj in range(rows.shape[1]):
            star_img, star_row0, star_col0 = APL.get_psf_image(
                rows[ii, jj],
                cols[ii, jj],
                norm=1.0,
                pix_zero_loc="edge",
                aca_image=False,
            )
            if star_row0 != -4 or star_col0 != -4:
                raise ValueError
            imgs[ii, jj] = star_img
    return imgs


# Could write this to a file but it only takes 18 ms
PSF_IMGS_GRID = make_psf_images(51)


@numba.njit()
def calc_legacy_flight_bgd(img: np.ndarray[np.float64]):
    """Compute the legacy flight background algorithm.

    a) Compute the average signal from pixels A1, B1, G1, H1, I4, J4, O4, and P4 in
       Figure 1-9.

    b) Compare each of these pixel signals with the average. If any differ from the
       average by more than ±150 percent or 10 A/D counts, whichever is larger, discard
       the one with the maximum deviation, or any one with the maximum deviation in case
       of a tie.

    c) Repeat steps (a) and (b), with discarded pixels omitted, until no pixels are
       discarded at step (b). Then report the final result in the aspect telemetry data.

    Typical execution time is around 240 ns. For 8 images and 1000 calls => 1.9 ms total.

    Parameters
    ----------
    img : np.ndarray[np.float64]
        The input image, which is expected to be 8x8 in units of e-/s and have the
        background pixels at the positions defined by ROW_BGD and COL_BGD.

    Returns
    -------
    float
        The computed background value.
    """
    keep = np.ones(8)
    vals = np.zeros(8, dtype=np.float64)
    for ii in range(8):
        vals[ii] = img[ROW_BGD[ii], COL_BGD[ii]]
    n_keep = 8

    while True:
        avg = 0.0
        for ii in range(8):
            if keep[ii]:
                avg += vals[ii]
        avg /= n_keep

        clip_limit = max(avg * 1.5, CLIP_MIN_BGD)
        max_dev = -1.0
        imax = -1
        for ii in range(8):
            if keep[ii]:
                dev = abs(vals[ii] - avg)
                if dev > max_dev:
                    max_dev = dev
                    imax = ii

        if max_dev > clip_limit:
            max_val = vals[imax]
            for ii in range(8):
                if vals[ii] == max_val and n_keep > 1:
                    keep[ii] = False
                    n_keep -= 1
        else:
            return avg


@numba.njit()
def centroid_fm(img: np.ndarray, bgd_est: float | np.ndarray):
    """
    First moment centroid of ``img``.

    Return FM centroid of image relative to the exact center of the 8x8 image.

    Parameters
    ----------
    img : np.ndarray
        Image including sources and background (e-/s) as 8x8 ndarray
    bgd_est : np.ndarray
        Background to subtract (e-/s) as 8x8 ndarray

    Returns
    -------
    row : float
        Row centroid in pixel coordinates
    col : float
        Column centroid in pixel coordinates
    norm : float
        Total flux in the image (e-/s) after background subtraction
    """
    norm = 0.0
    cent_row_sum = 0.0
    cent_col_sum = 0.0

    if isinstance(bgd_est, float):
        for ii, jj in zip(MOUSE_BIT_ROWS, MOUSE_BIT_COLS):
            img_bgd_sub = img[ii, jj] - bgd_est
            norm += img_bgd_sub
            cent_row_sum += ii * img_bgd_sub
            cent_col_sum += jj * img_bgd_sub
    else:
        for ii, jj in zip(MOUSE_BIT_ROWS, MOUSE_BIT_COLS):
            img_bgd_sub = img[ii, jj] - bgd_est[ii, jj]
            norm += img_bgd_sub
            cent_row_sum += ii * img_bgd_sub
            cent_col_sum += jj * img_bgd_sub

    if norm < 10.0:
        norm = 10.0

    # Compute centroids and convert to "edge" pixel convention
    cent_row = cent_row_sum / norm + 0.5
    cent_col = cent_col_sum / norm + 0.5

    return cent_row, cent_col, norm


@numba.njit()
def get_psf_image_from_grid(row: float, col: float, norm: float, img: np.ndarray):
    """Get PSF image from grid for given row/col and normalize."""
    # Find the nearest integer row/col and the fractional part. Need to use floor and
    # not round because round goes up for e.g. 25.5 and down for 24.5.
    row_int = int(np.floor(row + 0.5))
    col_int = int(np.floor(col + 0.5))
    row -= row_int
    col -= col_int
    n_rc = PSF_IMGS_GRID.shape[0]
    row_idx = int(round((row + 0.5) * (n_rc - 1)))
    col_idx = int(round((col + 0.5) * (n_rc - 1)))
    for ii in range(8):
        for jj in range(8):
            img[ii, jj] = PSF_IMGS_GRID[row_idx, col_idx, ii, jj] * norm
    return row_int - 4, col_int - 4


@numba.njit()
def clip(val: int, low: int, high: int):
    if val < low:
        return low
    elif val > high:
        return high
    else:
        return val


@numba.njit()
def shine_star_image(
    img: np.ndarray,
    img_row0: int,
    img_col0: int,
    star_row: float,
    star_col: float,
    star_norm: float,
    star_img: np.ndarray,
):
    star_row -= img_row0
    star_col -= img_col0
    star_row0, star_col0 = get_psf_image_from_grid(
        star_row, star_col, star_norm, star_img
    )
    row0 = clip(star_row0, 0, 8)
    row1 = clip(star_row0 + 8, 0, 8)
    col0 = clip(star_col0, 0, 8)
    col1 = clip(star_col0 + 8, 0, 8)
    img[row0:row1, col0:col1] += star_img[
        row0 - star_row0 : row1 - star_row0, col0 - star_col0 : col1 - star_col0
    ]


def star_track_numba(guide, dither_rs, dither_cs, dark: np.ndarray, stars):
    # Find all stars with centroid within a 9-pixel halfw box of guide
    # Note pix_zero_loc = 'edge' for all these.
    guide_row_cat = guide["row"]
    guide_col_cat = guide["col"]
    ok = (np.abs(stars["row"] - guide_row_cat) < 9) & (
        np.abs(stars["col"] - guide_col_cat) < 9
    )
    star_row0s = stars["row"][ok]
    star_col0s = stars["col"][ok]
    star_norms = transform.mag_to_count_rate(stars["mag"][ok])
    # print(star_norms)
    star_img = np.empty((8, 8), dtype=float)

    img_row = guide_row_cat
    img_col = guide_col_cat

    # Initial rate
    rate_row = 0.0
    rate_col = 0.0

    n_sim = len(dither_rs)
    cent_rows = np.zeros(n_sim, dtype=np.float64)
    cent_cols = np.zeros_like(dither_rs, dtype=np.float64)
    star_rows = np.zeros(n_sim, dtype=np.float64)
    star_cols = np.zeros_like(dither_rs, dtype=np.float64)
    norms = np.zeros_like(dither_rs, dtype=np.float64)
    img_row0s = np.zeros_like(dither_rs, dtype=np.int32)
    img_col0s = np.zeros_like(dither_rs, dtype=np.int32)

    for idx, dither_r, dither_c in zip(itertools.count(), dither_rs, dither_cs):
        # Next image location center as floats
        img_row = clip(img_row + rate_row, -508.0, 508.0)
        img_col = clip(img_col + rate_col, -508.0, 508.0)

        # Image readout lower left corner
        img_row0 = int(round(img_row)) - 4
        img_col0 = int(round(img_col)) - 4
        img_row0s[idx] = img_row0
        img_col0s[idx] = img_col0

        img = dark[
            img_row0 + 512 : img_row0 + 512 + 8, img_col0 + 512 : img_col0 + 512 + 8
        ].copy()

        # Shine star images onto img
        for star_row0, star_col0, star_norm in zip(star_row0s, star_col0s, star_norms):
            star_row = star_row0 + dither_r
            star_col = star_col0 + dither_c
            shine_star_image(
                img, img_row0, img_col0, star_row, star_col, star_norm, star_img
            )

        star_rows[idx] = guide_row_cat + dither_r
        star_cols[idx] = guide_col_cat + dither_c

        # bgd = calc_legacy_flight_bgd(np.asarray(img, dtype=np.float64))
        bgd = 30.0
        cent_row0, cent_col0, cent_norm = centroid_fm(img, bgd)
        cent_rows[idx] = cent_row0 + img_row0
        cent_cols[idx] = cent_col0 + img_col0
        norms[idx] = cent_norm

        rate_row = cent_rows[idx] - img_row
        rate_col = cent_cols[idx] - img_col

    out = {
        "star_row": star_rows,
        "star_col": star_cols,
        "cent_row": cent_rows,
        "cent_col": cent_cols,
        "norm": norms,
        "img_row0": img_row0s,
        "img_col0": img_col0s,
    }

    return apt.Table(out)


def run_aki_from_sim_obs(obsid, duration=None):
    from annie import sim_obs

    ao = sim_obs.AnnieObservation(obsid, duration)
    duration = ao.duration
    dither = ao.obs.dither
    dark = get_dark_cal_image(ao.obs.start, select="nearest", t_ccd_ref=ao.obs.t_ccd)

    dt = 2.05
    n_read = int(duration // dt)
    times = np.arange(n_read) * dt

    # pitch <=> col, yaw <=> row
    period_r = dither.yaw_period
    period_c = dither.pitch_period
    phase_r = dither.yaw_phase
    phase_c = dither.pitch_phase
    ampl_r = dither.yaw_ampl / 5.0
    ampl_c = dither.pitch_ampl / 5.0
    omega_r = 2 * np.pi / period_r
    omega_c = 2 * np.pi / period_c
    dither_rs = ampl_r * np.sin(omega_r * times + phase_r)
    dither_cs = ampl_c * np.sin(omega_c * times + phase_c)

    att_targ = ao.obs.att_targ
    stars = agasc.get_agasc_cone(att_targ.ra, att_targ.dec, 1.4)
    stars_yag, stars_zag = transform.radec_to_yagzag(
        stars["RA_PMCORR"], stars["DEC_PMCORR"], att_targ
    )
    stars["row"], stars["col"] = transform.yagzag_to_pixels(stars_yag, stars_zag)
    stars["mag"] = stars["MAG_ACA"]

    starcat = ao.obs.starcat
    ok = np.isin(starcat["type"], ["BOT", "GUI"])
    guides = starcat[ok]
    guides["row"], guides["col"] = transform.yagzag_to_pixels(
        guides["yang"],
        guides["zang"],
        t_aca=ao.obs.t_ccd + 41,
    )

    sdrs = {}
    for guide in guides:
        sdr = star_track_numba(guide, dither_rs, dither_cs, dark=dark, stars=stars)
        sdrs[guide["slot"]] = sdr

    return sdrs, ao
