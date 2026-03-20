import numba
import numpy as np

# Row and col positions for legacy flight background for calc_legacy_flight_bgd.
# Performance note: keeping this outside the function is faster.
ROW_BGD = np.array([0, 0, 0, 0, 7, 7, 7, 7])
COL_BGD = np.array([0, 1, 6, 7, 0, 1, 6, 7])
CLIP_MIN_BGD = 10 * 5 / 1.696  # 10 DN in e-/s for 1.696 second integration


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
