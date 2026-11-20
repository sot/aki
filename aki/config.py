"""Configuration objects for aki simulations.

A simulation is described by a ``SimConfig`` made of one small frozen dataclass per
concern:

- ``dither``: how the star moves on the CCD (``SinusoidDither``).
- ``dark``: the true dark current on the CCD (``DarkStatic``).
- ``bgd``: the background estimate the PEA subtracts before centroiding
  (``BgdLegacy``, ``BgdStatic`` or ``BgdDynamic``).

The class chooses the behavior, and each class holds only the options that apply to
it. This replaces annie-style ``*_source`` / ``*_data`` keyword pairs. Configs never
enter numba code: each one translates itself into plain arrays and integer mode codes
that ``aki.simulate`` passes to the tracking kernel.

Example::

    >>> from aki import SimConfig, SinusoidDither, DarkStatic, simulate
    >>> cfg = SimConfig(
    ...     duration=5000.0,
    ...     dither=SinusoidDither(yaw_ampl=16, pitch_ampl=16),
    ...     dark=DarkStatic.from_pixels({(0, 0): 1000.0}),
    ... )

Use ``dataclasses.replace`` to vary one option::

    >>> from dataclasses import replace
    >>> cfg_known = replace(cfg, bgd=BgdStatic(image=cfg.dark.image))
"""

from dataclasses import dataclass, field, fields
from typing import ClassVar

import numpy as np

__all__ = [
    "BGD_MODE_DYNAMIC",
    "BGD_MODE_IMAGE",
    "BGD_MODE_LEGACY",
    "CCD_SHAPE",
    "BgdDynamic",
    "BgdLegacy",
    "BgdStatic",
    "DarkStatic",
    "SimConfig",
    "SinusoidDither",
]

# Full-CCD images are indexed [row + 512, col + 512] where (row, col) is the integer
# pixel in edge coordinates (pix_zero_loc='edge').
CCD_SHAPE = (1024, 1024)

# Background mode codes passed to the numba tracking kernel
BGD_MODE_LEGACY = 0
BGD_MODE_IMAGE = 1
BGD_MODE_DYNAMIC = 2

# 10 DN in e-/s for a 1.696 s integration at 5 e-/DN (initial dynamic bgd stack value)
_DYN_BGD_STACK_INIT = 10 * 5 / 1.696

_EMPTY_IMAGE = np.zeros((0, 0), dtype=np.float64)
_EMPTY_IMAGE.flags.writeable = False


def _as_readonly_ccd_image(image, name: str) -> np.ndarray:
    """Return a read-only float64 copy of ``image`` after checking its shape."""
    arr = np.array(image, dtype=np.float64)
    if arr.shape != CCD_SHAPE:
        raise ValueError(f"{name} must have shape {CCD_SHAPE}, got {arr.shape}")
    arr.flags.writeable = False
    return arr


def _ccd_image_repr(image: np.ndarray) -> str:
    """Short summary of a CCD image for config reprs."""
    n_nonzero = np.count_nonzero(image)
    return f"<{image.shape[0]}x{image.shape[1]} max={image.max():.1f} n_nonzero={n_nonzero}>"


def _ccd_image_from_pixels(pixels: dict, background: float) -> np.ndarray:
    """Make a full-CCD image of ``background`` with ``pixels`` set."""
    image = np.full(CCD_SHAPE, float(background))
    for (row, col), val in pixels.items():
        image[int(row) + 512, int(col) + 512] = val
    return image


@dataclass(frozen=True, kw_only=True)
class SinusoidDither:
    """Ideal sinusoidal dither.

    Yaw maps to CCD row and pitch to CCD column. Amplitudes are in arcsec, periods in
    seconds and phases in radians. Defaults are 8 arcsec with yaw / pitch periods of
    1000 s / 707 s.
    """

    yaw_ampl: float = 8.0
    pitch_ampl: float = 8.0
    yaw_period: float = 1000.0
    pitch_period: float = 707.0
    yaw_phase: float = 0.0
    pitch_phase: float = 0.0

    def __post_init__(self):
        for fld in fields(self):
            object.__setattr__(self, fld.name, float(getattr(self, fld.name)))
        if self.yaw_period <= 0 or self.pitch_period <= 0:
            raise ValueError("dither periods must be positive")

    def get_offsets(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row and column dither offsets (pixels) at ``times`` (seconds)."""
        from aki.aki import get_pure_dither_samples

        return get_pure_dither_samples(self, times)


@dataclass(frozen=True, kw_only=True, eq=False)
class DarkStatic:
    """True dark current that does not change with time.

    Parameters
    ----------
    image : array-like
        Full-CCD dark current image (e-/s), shape (1024, 1024), indexed
        ``[row + 512, col + 512]``. A read-only copy is stored.
    """

    image: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "image", _as_readonly_ccd_image(self.image, "image"))

    def __repr__(self):
        return f"DarkStatic(image={_ccd_image_repr(self.image)})"

    @classmethod
    def from_pixels(cls, pixels: dict, background: float = 0.0) -> "DarkStatic":
        """Uniform dark current ``background`` with some pixels set.

        Parameters
        ----------
        pixels : dict
            Mapping ``{(row, col): dark_current}`` with integer pixel coordinates in
            edge convention, e.g. ``{(0, 0): 1000.0}`` sets the pixel spanning rows
            0-1 and columns 0-1.
        background : float
            Dark current (e-/s) for all other pixels.
        """
        return cls(image=_ccd_image_from_pixels(pixels, background))

    def get_image(self) -> np.ndarray:
        """Full-CCD true dark current image (e-/s)."""
        return self.image


@dataclass(frozen=True, kw_only=True)
class BgdLegacy:
    """Legacy flight background: one value per readout from the 8 corner pixels.

    See ``aki.calc_legacy_flight_bgd``.
    """

    mode: ClassVar[int] = BGD_MODE_LEGACY
    # Placeholder so the tracking kernel always gets a float64 2-d array
    image: ClassVar[np.ndarray] = _EMPTY_IMAGE


@dataclass(frozen=True, kw_only=True, eq=False)
class BgdStatic:
    """Fixed per-pixel background image subtracted from every readout.

    With ``image`` equal to the true dark current this is a perfectly known
    background. A SamBa background is the same kind of fixed image, built from
    sampled scans.

    Parameters
    ----------
    image : array-like
        Full-CCD background image (e-/s), shape (1024, 1024), indexed
        ``[row + 512, col + 512]``. A read-only copy is stored.
    """

    mode: ClassVar[int] = BGD_MODE_IMAGE
    image: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "image", _as_readonly_ccd_image(self.image, "image"))

    def __repr__(self):
        return f"BgdStatic(image={_ccd_image_repr(self.image)})"

    @classmethod
    def from_pixels(cls, pixels: dict, background: float = 0.0) -> "BgdStatic":
        """Uniform background ``background`` with some pixels set.

        See ``DarkStatic.from_pixels``.
        """
        return cls(image=_ccd_image_from_pixels(pixels, background))


@dataclass(frozen=True, kw_only=True)
class BgdDynamic:
    """Dynamic background (annie ``DynamicBgd2``).

    Each tracked readout writes the 32 pixels outside the 6x6 centroid region (the
    8x8 edge ring plus the four 6x6 corners) into one layer of a 5-deep stack of
    16x16 CCD-fixed images that follows the readout window. The background for each
    of the 8x8 pixels is the median of the 5 stack values at that CCD pixel. The
    estimate is then capped at the image value to prevent over-subtraction.

    If the minimum of those 32 pixels exceeds ``outer_min_thresh`` (e.g. a bright
    nearby star fills the window), the stack is left untouched and the flat legacy
    flight background is used for that readout. The stack is cleared when tracking
    is reset after a guide star loss, like annie's default
    ``clear_bgd_stack_on_search=True``.

    Parameters
    ----------
    outer_min_thresh : float
        Threshold (e-/s) on the minimum outer pixel value above which the legacy
        flat background is used. Default 500.
    use_outer_min : bool
        If True, floor the dynamic background at the minimum outer pixel value.
        Default False.
    stack_init : float
        Initial value (e-/s) for stack pixels that have not been sampled. Default
        is 10 DN (about 29.5 e-/s).
    """

    mode: ClassVar[int] = BGD_MODE_DYNAMIC
    # Placeholder so the tracking kernel always gets a float64 2-d array
    image: ClassVar[np.ndarray] = _EMPTY_IMAGE

    outer_min_thresh: float = 500.0
    use_outer_min: bool = False
    stack_init: float = _DYN_BGD_STACK_INIT

    def __post_init__(self):
        object.__setattr__(self, "outer_min_thresh", float(self.outer_min_thresh))
        object.__setattr__(self, "use_outer_min", bool(self.use_outer_min))
        object.__setattr__(self, "stack_init", float(self.stack_init))


@dataclass(frozen=True, kw_only=True, eq=False)
class SimConfig:
    """Full configuration for an aki single-star tracking simulation.

    Parameters
    ----------
    dark : DarkStatic
        True dark current on the CCD. Required.
    duration : float
        Simulated time (seconds). Readouts are every ``aki.DT_READOUT`` seconds
        for times in ``[0, duration)``.
    dither : SinusoidDither
        Dither pattern.
    bgd : BgdLegacy, BgdStatic or BgdDynamic
        Background estimate subtracted before centroiding.
    """

    dark: DarkStatic
    duration: float = 1000.0
    dither: SinusoidDither = field(default_factory=SinusoidDither)
    bgd: BgdLegacy | BgdStatic | BgdDynamic = field(default_factory=BgdLegacy)

    def __post_init__(self):
        object.__setattr__(self, "duration", float(self.duration))
        if self.duration <= 0:
            raise ValueError("duration must be positive")
