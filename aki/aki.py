import numba
import numpy as np

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
def centroid_fm(img: np.ndarray, bgd_est: np.ndarray):
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
