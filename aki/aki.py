import itertools

import agasc
import astropy.table as apt
import numba
import numpy as np
from chandra_aca import transform
from chandra_aca.aca_image import AcaPsfLibrary
from cxotime import CxoTime
from mica.archive.aca_dark import get_dark_cal_image
from ska_trend.centroid_dashboard import app as cent_app

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
    """Look up an 8x8 PSF image from ``PSF_IMGS_GRID`` for a given sub-pixel position.

    The PSF grid is indexed by fractional row/col offset from the nearest integer
    pixel (see ``make_psf_images``), so this splits ``row``/``col`` into an integer
    part (the image lower-left corner offset) and a fractional part used to select
    the nearest pre-computed PSF sample, which is then scaled by ``norm``.

    Parameters
    ----------
    row : float
        Star row position in pixel coordinates.
    col : float
        Star column position in pixel coordinates.
    norm : float
        Total flux to scale the PSF image by (e-/s).
    img : np.ndarray
        8x8 output array that is filled in-place with the scaled PSF image.

    Returns
    -------
    row0 : int
        Row offset of the image lower-left corner relative to ``row``.
    col0 : int
        Column offset of the image lower-left corner relative to ``col``.
    """
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
    """Clip ``val`` to the closed interval ``[low, high]``.

    Parameters
    ----------
    val : int
        Value to clip.
    low : int
        Lower bound.
    high : int
        Upper bound.

    Returns
    -------
    int
        ``val`` clamped to ``[low, high]``.
    """
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
    """Add a star's PSF image onto ``img`` in-place at the appropriate location.

    Looks up the PSF for the star's position relative to the image's lower-left
    corner (``img_row0``, ``img_col0``) via ``get_psf_image_from_grid``, then adds
    the (possibly clipped, if it extends beyond the 8x8 ``img`` bounds) overlapping
    portion of that PSF onto ``img``.

    Parameters
    ----------
    img : np.ndarray
        8x8 image array to add the star's PSF onto, modified in-place.
    img_row0 : int
        Row of the lower-left corner of ``img`` in absolute pixel coordinates.
    img_col0 : int
        Column of the lower-left corner of ``img`` in absolute pixel coordinates.
    star_row : float
        Star row position in absolute pixel coordinates.
    star_col : float
        Star column position in absolute pixel coordinates.
    star_norm : float
        Total star flux (e-/s) used to scale the PSF image.
    star_img : np.ndarray
        8x8 scratch array used to hold the star's PSF image, overwritten in-place.
    """
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


@numba.njit()
def _advance_readout(
    dither_r,
    dither_c,
    guide_row_cat,
    guide_col_cat,
    star_row0s,
    star_col0s,
    star_norms,
    star_img,
    dark,
    min_img_sum,
    img_row,
    img_col,
    rate_row,
    rate_col,
    gs_loss_count,
    img_track,
):
    """Advance guide-star tracking by one simulated readout.

    Synthesizes the 8x8 readout image at the current tracked position, adds
    nearby star PSFs, computes a first-moment centroid, and updates the
    tracking rate and GS-loss state for the next readout.

    Parameters
    ----------
    dither_r, dither_c : float
        Dither offset (pixels) at this readout.
    guide_row_cat, guide_col_cat : float
        Guide star catalog position (pixels), before dither.
    star_row0s, star_col0s, star_norms : np.ndarray
        Catalog positions (pixels) and count rates (e-/s) of candidate stars
        near the guide star.
    star_img : np.ndarray
        8x8 scratch array used to hold a star's PSF image.
    dark : np.ndarray
        Full dark current CCD image (e-/s) used as the image background.
    min_img_sum : float
        Image flux (e-/s) threshold below which tracking is dropped.
    img_row, img_col, rate_row, rate_col, gs_loss_count, img_track
        Tracking state carried over from the previous readout.

    Returns
    -------
    cent_row, cent_col : float
        Computed centroid in absolute pixel coordinates.
    img_sum : float
        Background-subtracted image flux (e-/s).
    img_row0, img_col0 : int
        Readout image lower-left corner in absolute pixel coordinates.
    star_row, star_col : float
        True (dithered) guide star position.
    img_row, img_col, rate_row, rate_col, gs_loss_count, img_track
        Updated tracking state for the next readout.
    """
    # Next image location center as floats
    img_row = clip(img_row + rate_row, -508.0, 508.0)
    img_col = clip(img_col + rate_col, -508.0, 508.0)

    # Image readout lower left corner
    img_row0 = int(round(img_row)) - 4
    img_col0 = int(round(img_col)) - 4

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

    guide_row = guide_row_cat + dither_r
    guide_col = guide_col_cat + dither_c

    # bgd = calc_legacy_flight_bgd(np.asarray(img, dtype=np.float64))
    bgd = 30.0
    # Centroid row/col relative to lower left pixel edge at 0, 0
    cent_row0, cent_col0, img_sum = centroid_fm(img, bgd)
    # Centroid row/col in absolute coordinates (with 0, 0 at the CCD center)
    cent_row = cent_row0 + img_row0
    cent_col = cent_col0 + img_col0

    if img_sum < min_img_sum:
        img_track = False

    if not img_track:
        # Corresponds to RACQ state with star below threshold and not tracking.
        rate_row = 0.0
        rate_col = 0.0
    else:
        rate_row = cent_row - img_row
        rate_col = cent_col - img_col

    if (
        not img_track
        or abs(guide_row - cent_row) > 1.0  # 1 pixel = 5 arcsec
        or abs(guide_col - cent_col) > 1.0
    ):
        gs_loss_count += 1

    # PCAD GS_loss_count threshold is 100 readouts at 1.025 s/readout. Here we
    # simulate only each 2.05 s image read.
    if gs_loss_count > 50:
        gs_loss_count = 0
        rate_row = 0.0
        rate_col = 0.0
        img_row = guide_row
        img_col = guide_col

    return (
        cent_row,
        cent_col,
        img_sum,
        img_row0,
        img_col0,
        guide_row,
        guide_col,
        img_row,
        img_col,
        rate_row,
        rate_col,
        gs_loss_count,
        img_track,
    )


def star_track_numba(guide, dither_rs, dither_cs, dark: np.ndarray, stars):
    """Simulate ACA image-readout tracking of a single guide star over time.

    For each dither offset in ``dither_rs``/``dither_cs``, this synthesizes an 8x8
    ACA readout image (dark current plus PSF images of nearby stars from ``stars``),
    computes a first-moment centroid, and updates the tracked image position for the
    next readout using a simple rate-based tracking loop. Tracking is dropped if the
    image sum falls below a minimum threshold or if the guide star loss counter (GS
    loss) exceeds its limit, in which case tracking resets to the catalog position.

    Parameters
    ----------
    guide : dict-like
        Guide star catalog entry with fields ``row``, ``col``, ``maxmag``, and
        ``slot``, in pixel coordinates (``pix_zero_loc='edge'``).
    dither_rs : np.ndarray
        Dither offset in row (pixels) at each simulated readout.
    dither_cs : np.ndarray
        Dither offset in column (pixels) at each simulated readout.
    dark : np.ndarray
        Full dark current CCD image (e-/s) used as the image background.
    stars : dict-like / table
        Candidate stars near the guide star, with fields ``row``, ``col``, ``mag``.

    Returns
    -------
    dict
        Dictionary of per-readout arrays with keys ``time``, ``star_row``,
        ``star_col`` (true guide star position including dither), ``cent_row``,
        ``cent_col`` (computed centroid), ``img_sum``, ``img_row0``, ``img_col0``
        (readout image lower-left corner), and ``img_track`` (bool tracking state).
    """
    # Find all stars with centroid within a 9-pixel halfw box of guide
    # Note pix_zero_loc = 'edge' for all these.
    guide_row_cat = float(guide["row"])
    guide_col_cat = float(guide["col"])
    ok = (np.abs(stars["row"] - guide_row_cat) < 9) & (
        np.abs(stars["col"] - guide_col_cat) < 9
    )
    star_row0s = np.asarray(stars["row"][ok], dtype=np.float64)
    star_col0s = np.asarray(stars["col"][ok], dtype=np.float64)
    star_norms = np.asarray(
        transform.mag_to_count_rate(stars["mag"][ok]), dtype=np.float64
    )
    # print(star_norms)
    star_img = np.empty((8, 8), dtype=float)

    # Below this threshold stop tracking
    min_img_sum = float(transform.mag_to_count_rate(guide["maxmag"])) / 2

    img_row = guide_row_cat
    img_col = guide_col_cat

    # Initial rate
    rate_row = 0.0
    rate_col = 0.0

    gs_loss_count = 0
    img_track = True

    n_sim = len(dither_rs)
    cent_rows = np.zeros(n_sim, dtype=np.float64)
    cent_cols = np.zeros(n_sim, dtype=np.float64)
    star_rows = np.zeros(n_sim, dtype=np.float64)
    star_cols = np.zeros(n_sim, dtype=np.float64)
    img_sums = np.zeros(n_sim, dtype=np.float64)
    img_row0s = np.zeros(n_sim, dtype=np.int32)
    img_col0s = np.zeros(n_sim, dtype=np.int32)
    img_tracks = np.zeros(n_sim, dtype=bool)

    for idx, dither_r, dither_c in zip(itertools.count(), dither_rs, dither_cs):
        (
            cent_rows[idx],
            cent_cols[idx],
            img_sums[idx],
            img_row0s[idx],
            img_col0s[idx],
            star_rows[idx],
            star_cols[idx],
            img_row,
            img_col,
            rate_row,
            rate_col,
            gs_loss_count,
            img_track,
        ) = _advance_readout(
            dither_r,
            dither_c,
            guide_row_cat,
            guide_col_cat,
            star_row0s,
            star_col0s,
            star_norms,
            star_img,
            dark,
            min_img_sum,
            img_row,
            img_col,
            rate_row,
            rate_col,
            gs_loss_count,
            img_track,
        )
        img_tracks[idx] = img_track

    out = {
        "time": np.arange(n_sim) * 2.05,
        "star_row": star_rows,
        "star_col": star_cols,
        "cent_row": cent_rows,
        "cent_col": cent_cols,
        "img_sum": img_sums,
        "img_row0": img_row0s,
        "img_col0": img_col0s,
        "img_track": img_tracks,
    }

    return out


def run_aki_from_sim_obs(obsid, duration=None, dither_source="sinusoid"):
    """Run the guide star tracking simulation for a simulated observation (obsid).

    Builds a simulated observation via ``annie.sim_obs.AnnieObservation``, derives
    the dither motion, dark current image, and candidate/guide star catalogs for
    that observation, then runs ``star_track_numba`` for each guide star and wraps
    the results in ``CentroidResidualsLite`` objects for comparison against the
    true (dithered) star positions.

    Parameters
    ----------
    obsid : int
        Observation ID to simulate.
    duration : float, optional
        Observation duration in seconds. If not given, uses the full duration from
        the simulated observation.
    dither_source : str, optional
        Source of the dither motion, either ``"sinusoid"`` (default) to generate
        ideal sinusoidal dither from the commanded dither parameters, or
        ``"flight-att"`` to derive the dither from the flight attitude telemetry
        (``AOATTQT``) relative to the target attitude.

    Returns
    -------
    sdrs : dict
        Per-slot dictionary of simulated tracking results, keyed by guide star slot
        (see ``star_track_numba`` return value for the per-slot dict contents).
    crs_sim : dict
        Per-slot dictionary of ``CentroidResidualsLite`` objects built from the
        centroid residuals (dyag/dzag) in ``sdrs``.
    ao : annie.sim_obs.AnnieObservation
        The simulated observation object.
    """
    from annie import sim_obs

    if dither_source not in ("sinusoid", "flight-att"):
        raise ValueError(
            f"dither_source must be 'sinusoid' or 'flight-att', not {dither_source!r}"
        )

    ao = sim_obs.AnnieObservation(obsid, duration)
    duration = ao.duration
    dither = ao.obs.dither
    dark = get_dark_cal_image(ao.obs.start, select="nearest", t_ccd_ref=ao.obs.t_ccd)

    dt = 2.05
    n_read = int(duration // dt)
    times = np.arange(n_read) * dt

    # pitch <=> col, yaw <=> row
    if dither_source == "sinusoid":
        dither_rs, dither_cs = get_pure_dither_samples(dither, times)
    else:
        dither_rs, dither_cs = get_flight_att_dither_samples(ao, times)

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

    crs_sim = {}
    for slot, sdr in sdrs.items():
        crs_sim[slot] = cent_app.CentroidResidualsLite(
            dyags=(sdr["cent_row"] - sdr["star_row"]) * 5.0,
            dzags=(sdr["cent_col"] - sdr["star_col"]) * 5.0,
            yag_times=sdr["time"],
            zag_times=sdr["time"],
        )

    return sdrs, crs_sim, ao


def get_pure_dither_samples(dither, times):
    """Compute sinusoidal dither offsets for row and column at specified times.

    Converts dither parameters (periods, phases, amplitudes) to sinusoidal motion
    offsets in pixel coordinates. The dither motion is used to move the tracked
    image around the ACA detector during science observations.

    Parameters
    ----------
    dither : dict-like
        Dither motion parameters with fields ``yaw_period``, ``pitch_period``,
        ``yaw_phase``, ``pitch_phase``, ``yaw_ampl``, and ``pitch_ampl``. Periods
        are in seconds, phases in radians, and amplitudes in arcseconds.
    times : np.ndarray
        Time values (seconds) at which to compute dither offsets.

    Returns
    -------
    dither_rs : np.ndarray
        Dither offset in row (pixels) at each time, computed as sinusoidal motion
        from yaw (pitch in spacecraft coordinates).
    dither_cs : np.ndarray
        Dither offset in column (pixels) at each time, computed as sinusoidal
        motion from pitch (yaw in spacecraft coordinates).
    """
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
    return dither_rs, dither_cs


def get_flight_att_dither_samples(ao, times):
    """Compute dither offsets for row and column from flight attitude telemetry.

    Takes the delta quaternion between the target attitude and the on-board
    estimated attitude (``AOATTQT``) and converts the resulting pitch and yaw
    offsets to pixel offsets, interpolated onto ``times``. Unlike
    ``get_pure_dither_samples``, this captures the actual attitude motion
    including dither, drift, and any attitude control residuals.

    Parameters
    ----------
    ao : annie.sim_obs.AnnieObservation
        Simulated observation, providing the target attitude
        (``ao.obs.att_targ``), the attitude telemetry (``ao.obs.aoattqt``), and
        the observation start time (``ao.obs.start``).
    times : np.ndarray
        Time values (seconds relative to ``ao.obs.start``) at which to compute
        dither offsets. These must fall within the attitude telemetry coverage,
        apart from an edge tolerance of one telemetry sample interval.

    Returns
    -------
    dither_rs : np.ndarray
        Dither offset in row (pixels) at each time, from the yaw offset.
    dither_cs : np.ndarray
        Dither offset in column (pixels) at each time, from the pitch offset.

    Raises
    ------
    ValueError
        If ``times`` extends beyond the attitude telemetry coverage by more than
        one telemetry sample interval. Interpolation would otherwise silently
        clip to the endpoint value, giving a constant (non-dithering) offset.
    """
    aoattqt = ao.obs.aoattqt
    dq = ao.obs.att_targ.dq(aoattqt.vals)

    # Telemetry times are CXC seconds while ``times`` is relative to obs start.
    att_times = aoattqt.times - CxoTime(ao.obs.start).secs

    # Telemetry does not exactly bracket the obs start/stop, so tolerate a gap of
    # one sample interval at each end but reject anything genuinely uncovered.
    tol = np.median(np.diff(att_times))
    if times[0] < att_times[0] - tol or times[-1] > att_times[-1] + tol:
        raise ValueError(
            f"times [{times[0]:.1f}, {times[-1]:.1f}] extend beyond AOATTQT "
            f"coverage [{att_times[0]:.1f}, {att_times[-1]:.1f}] "
            f"(tolerance {tol:.1f} s) relative to obs start {ao.obs.start}"
        )

    # Convert degrees to arcsec and then to pixels. pitch <=> col, yaw <=> row.
    dither_rs = np.interp(times, att_times, dq.yaw * 3600 / 5.0)
    dither_cs = np.interp(times, att_times, dq.pitch * 3600 / 5.0)

    return dither_rs, dither_cs
