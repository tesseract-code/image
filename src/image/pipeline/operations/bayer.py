import cv2
import numpy as np

try:
    from scipy.ndimage import convolve
except ImportError:
    def convolve(array, kernel):
        """Safe fallback for convolution without Scipy"""
        pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
        padded = np.pad(array, (
            (pad_h, pad_h), (pad_w, pad_w)
        ), mode='reflect')
        s = kernel.shape + tuple(np.subtract(padded.shape, kernel.shape) + 1)
        strd = np.lib.stride_tricks.as_strided
        sub_m = strd(padded, shape=s, strides=padded.strides * 2)
        return np.einsum('ij,ijkl->kl', kernel, sub_m)


def mosaic(rgb, pattern='RGGB'):
    """
    Simulates a Bayer CFA sensor capture (Reverse Demosaic).

    This function subsamples a full RGB image according to the specified
    Bayer pattern, effectively mimicking the raw data captured by a sensor.

    Optimization Note:
    Uses boolean masking for direct memory assignment (O(N)) rather than
    arithmetic multiplication (O(3N)), making it significantly faster
    and memory efficient.

    Parameters
    ----------
    rgb : ndarray
        (Height, Width, 3) RGB image.
    pattern : str
        Bayer pattern {'RGGB', 'BGGR', 'GRBG', 'GBRG'}.

    Returns
    -------
    ndarray
        (Height, Width) Raw Bayer array (2D).
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB array, got shape {rgb.shape}")

    h, w, _ = rgb.shape

    # Pre-allocate output array with same dtype as input (uint8/uint16/float)
    cfa = np.zeros((h, w), dtype=rgb.dtype)

    # Retrieve the boolean masks
    # (Reuses the existing high-performance mask generator)
    R_m, G_m, B_m = masks((h, w), pattern)

    # -------------------------------------------------------------------------
    # OPTIMIZATION: Direct Indexed Assignment
    # -------------------------------------------------------------------------
    # Instead of: cfa = R * R_m + G * G_m ... (which multiplies by 0 often)
    # We use:     cfa[mask] = channel[mask]
    # This acts as a sparse scatter operation.

    # 1. Fill Red pixels
    # We select the Red channel (..., 0) and filter by the Red Mask
    cfa[R_m] = rgb[..., 0][R_m]

    # 2. Fill Green pixels
    cfa[G_m] = rgb[..., 1][G_m]

    # 3. Fill Blue pixels
    cfa[B_m] = rgb[..., 2][B_m]

    return cfa


def masks(shape, pattern):
    """Internal helper for mask generation"""
    pattern = pattern.upper()
    channels = dict((channel, np.zeros(shape, dtype=bool)) for channel in 'RGB')
    offsets = {
        'RGGB': [('R', 0, 0), ('G', 0, 1), ('G', 1, 0), ('B', 1, 1)],
        'BGGR': [('B', 0, 0), ('G', 0, 1), ('G', 1, 0), ('R', 1, 1)],
        'GRBG': [('G', 0, 0), ('R', 0, 1), ('B', 1, 0), ('G', 1, 1)],
        'GBRG': [('G', 0, 0), ('B', 0, 1), ('R', 1, 0), ('G', 1, 1)],
    }
    for channel, y, x in offsets[pattern]:
        channels[channel][y::2, x::2] = 1
    return tuple(channels[c] for c in 'RGB')


def _demosaic_malvar_he_cutler(cfa, pattern):
    """
    Implementation of Malvar-He-Cutler (2004).
    Heavy math operations using 5x5 convolution kernels.
    """
    # Ensure float for precision math
    cfa_f = cfa.astype(np.float32)

    # Normalize if input is integer (assuming 8-bit or 16-bit)
    # This prevents overflow during convolution additions
    is_int = cfa.dtype.kind in 'ui'
    max_val = 255.0 if cfa.dtype == np.uint8 else 65535.0
    if is_int:
        cfa_f /= max_val

    R_m, G_m, B_m = masks(cfa.shape, pattern)

    # 5x5 Kernels (Sum=8)
    K_G = np.array(
        [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0],
         [0, 0, -1, 0, 0]]) / 8.0
    K_RB_diag = np.array(
        [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
         [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]]) / 8.0
    K_RB_axial = np.array(
        [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
         [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]]) / 8.0

    conv_G = convolve(cfa_f, K_G)
    conv_diag = convolve(cfa_f, K_RB_diag)
    conv_h = convolve(cfa_f, K_RB_axial)
    conv_v = convolve(cfa_f, K_RB_axial.T)

    G = np.where(G_m, cfa_f, conv_G)

    R_rows = np.any(R_m, axis=1)[:, np.newaxis]
    B_rows = np.any(B_m, axis=1)[:, np.newaxis]

    R = (cfa_f * R_m) + (conv_diag * B_m) + (conv_h * (G_m & R_rows)) + (
            conv_v * (G_m & ~R_rows))
    B = (cfa_f * B_m) + (conv_diag * R_m) + (conv_h * (G_m & B_rows)) + (
            conv_v * (G_m & ~B_rows))

    out = np.stack([R, G, B], axis=-1)

    # Restore original range if input was int
    if is_int:
        out = np.clip(out * max_val, 0, max_val).astype(cfa.dtype)

    return out


# ---------------------------------------------------------------------
# 2. OPTIMIZED WRAPPER (The Production API)
# ---------------------------------------------------------------------

def demosaic(cfa, pattern='RGGB', alg_type='FAST'):
    """
    Main entry point for demosaicing.

    Parameters
    ----------
    cfa : ndarray
        The raw bayer array (uint8 or uint16).
    pattern : str
        'RGGB', 'BGGR', 'GRBG', 'GBRG'
    alg_type : str
        'FAST'    -> Uses OpenCV (Bilinear). ~2ms.
        'QUALITY' -> Uses Malvar-He-Cutler (Numpy). ~200ms+.
                     Use this for Snapshots or Paused View.
    """
    pattern = pattern.upper()

    if alg_type == 'FAST':
        # ---------------------------------------------------------
        # OPTIMIZATION: Use OpenCV C++ implementation
        # ---------------------------------------------------------

        # Map string pattern to OpenCV constants
        # Note: OpenCV naming convention is COLOR_Bayer{StartPixel}2RGB
        cv_patterns = {
            'RGGB': cv2.COLOR_BayerRG2RGB,
            'BGGR': cv2.COLOR_BayerBG2RGB,
            'GRBG': cv2.COLOR_BayerGR2RGB,
            'GBRG': cv2.COLOR_BayerGB2RGB
        }

        if pattern not in cv_patterns:
            raise ValueError(f"Unknown pattern: {pattern}")

        # cv2.cvtColor is heavily optimized with SIMD/AVX2
        return cv2.cvtColor(cfa, cv_patterns[pattern])

    elif alg_type == 'QUALITY':
        # ---------------------------------------------------------
        # QUALITY: Use Malvar-He-Cutler
        # ---------------------------------------------------------
        return _demosaic_malvar_he_cutler(cfa, pattern)

    else:
        raise ValueError("alg_type must be 'FAST' or 'QUALITY'")
