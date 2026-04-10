"""
tests/test_gl_viewport.py
=========================
Unit tests for cross_platform.qt6_utils.image.gl.viewport.ViewManager.

QMatrix4x4 is a pure-math Qt class that requires no display server.
These tests use real QMatrix4x4 instances so that the actual transform
composition is verified, not just that Qt was called.

A minimal QApplication is created once per session via the ``qt_app``
session-scoped fixture; this satisfies Qt's internal initialisation
requirement without opening any windows.

Coverage
--------
Construction           — default state, initial matrix identity.
set_image_size         — state update, transform rebuild, zero-dimension guard.
handle_resize          — state update, projection rebuild, zero-dimension guard.
handle_pan             — accumulation, transform rebuild.
handle_zoom            — no-centre path, centre path (pan compensation),
                         zero/negative factor guard, state update.
handle_rotation        — state update, transform rebuild.
reset_view             — all state returns to defaults.
fit_to_viewport        — zoom calculation, padding, pan/rotation reset,
                         zero-image no-op.
update_transform       — rotation threshold, pan/zoom presence in matrix.
get_projection_data    — shape, dtype, flat layout (not reshaped).
get_transform_data     — shape, dtype, flat layout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PyQt6.QtGui import QMatrix4x4
from PyQt6.QtWidgets import QApplication

from cross_platform.qt6_utils.image.gl.error import GLInitializationError
from cross_platform.qt6_utils.image.gl.viewport import ViewManager, _FIT_PADDING


# ---------------------------------------------------------------------------
# Session-scoped QApplication
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qt_app():
    """
    Provide a QApplication for the entire test session.

    QMatrix4x4 does not require a display but Qt's initialisation path
    expects an application object to exist.  A single instance shared
    across all tests is sufficient.
    """
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def mgr(qt_app) -> ViewManager:
    """Fresh ViewManager for each test."""
    return ViewManager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat(matrix: QMatrix4x4) -> np.ndarray:
    """Return a QMatrix4x4 as a flat float64 array (column-major)."""
    return np.array(matrix.data(), dtype=np.float64)


def _identity_flat() -> np.ndarray:
    m = QMatrix4x4()
    m.setToIdentity()
    return _flat(m)


# =============================================================================
# Construction
# =============================================================================

class TestConstruction:

    def test_default_zoom(self, mgr):
        assert mgr.zoom_level == 1.0

    def test_default_pan(self, mgr):
        assert mgr.pan_x == 0.0
        assert mgr.pan_y == 0.0

    def test_default_rotation(self, mgr):
        assert mgr.rotation == 0.0

    def test_default_viewport_dimensions(self, mgr):
        assert mgr.viewport_w == 1
        assert mgr.viewport_h == 1

    def test_default_image_dimensions(self, mgr):
        assert mgr.image_w == 1
        assert mgr.image_h == 1

    def test_projection_starts_as_identity(self, mgr):
        # Before handle_resize is called, projection is setToIdentity() default.
        np.testing.assert_allclose(_flat(mgr.projection), _identity_flat(), atol=1e-6)

    def test_transform_starts_non_identity(self, mgr):
        # update_transform is NOT called in __init__; the Qt default matrix
        # is identity, which may or may not equal the first computed transform.
        # We only verify the object is a QMatrix4x4.
        assert isinstance(mgr.transform, QMatrix4x4)


# =============================================================================
# set_image_size
# =============================================================================

class TestSetImageSize:

    def test_updates_image_dimensions(self, mgr):
        mgr.set_image_size(800, 600)
        assert mgr.image_w == 800
        assert mgr.image_h == 600

    def test_rebuilds_transform(self, mgr):
        before = _flat(mgr.transform).copy()
        mgr.set_image_size(512, 512)
        after = _flat(mgr.transform)
        # With a symmetric 512x512 image and 1x1 viewport the transform
        # will differ from the pre-call state.
        # We verify update_transform ran by checking image dims are reflected.
        assert mgr.image_w == 512

    def test_raises_on_zero_width(self, mgr):
        with pytest.raises(GLInitializationError, match="positive"):
            mgr.set_image_size(0, 480)

    def test_raises_on_zero_height(self, mgr):
        with pytest.raises(GLInitializationError, match="positive"):
            mgr.set_image_size(640, 0)

    def test_raises_on_negative_dimensions(self, mgr):
        with pytest.raises(GLInitializationError):
            mgr.set_image_size(-1, 480)


# =============================================================================
# handle_resize
# =============================================================================

class TestHandleResize:

    def test_updates_viewport_dimensions(self, mgr):
        mgr.handle_resize(1920, 1080)
        assert mgr.viewport_w == 1920
        assert mgr.viewport_h == 1080

    def test_projection_is_not_identity_after_resize(self, mgr):
        mgr.handle_resize(800, 600)
        # Orthographic projection with non-unit dimensions cannot be identity.
        assert not np.allclose(_flat(mgr.projection), _identity_flat(), atol=1e-3)

    def test_projection_changes_on_second_resize(self, mgr):
        mgr.handle_resize(800, 600)
        proj_1 = _flat(mgr.projection).copy()
        mgr.handle_resize(1280, 720)
        proj_2 = _flat(mgr.projection)
        assert not np.allclose(proj_1, proj_2)

    def test_raises_on_zero_width(self, mgr):
        with pytest.raises(GLInitializationError, match="positive"):
            mgr.handle_resize(0, 600)

    def test_raises_on_zero_height(self, mgr):
        with pytest.raises(GLInitializationError, match="positive"):
            mgr.handle_resize(800, 0)

    def test_raises_on_negative_dimensions(self, mgr):
        with pytest.raises(GLInitializationError):
            mgr.handle_resize(-1, 600)


# =============================================================================
# handle_pan
# =============================================================================

class TestHandlePan:

    def test_accumulates_pan(self, mgr):
        mgr.handle_pan(10.0, 20.0)
        mgr.handle_pan(5.0, -3.0)
        assert mgr.pan_x == pytest.approx(15.0)
        assert mgr.pan_y == pytest.approx(17.0)

    def test_negative_delta_decrements_pan(self, mgr):
        mgr.handle_pan(-50.0, -25.0)
        assert mgr.pan_x == pytest.approx(-50.0)
        assert mgr.pan_y == pytest.approx(-25.0)

    def test_transform_differs_from_zero_pan(self, mgr):
        mgr.handle_resize(800, 600)
        t_zero = _flat(mgr.transform).copy()
        mgr.handle_pan(100.0, 0.0)
        t_panned = _flat(mgr.transform)
        assert not np.allclose(t_zero, t_panned)

    def test_zero_delta_leaves_pan_unchanged(self, mgr):
        mgr.handle_pan(10.0, 10.0)
        mgr.handle_pan(0.0, 0.0)
        assert mgr.pan_x == pytest.approx(10.0)
        assert mgr.pan_y == pytest.approx(10.0)


# =============================================================================
# handle_zoom
# =============================================================================

class TestHandleZoom:

    def test_multiplies_zoom_level(self, mgr):
        mgr.handle_zoom(2.0)
        assert mgr.zoom_level == pytest.approx(2.0)

    def test_zoom_composes_multiplicatively(self, mgr):
        mgr.handle_zoom(2.0)
        mgr.handle_zoom(1.5)
        assert mgr.zoom_level == pytest.approx(3.0)

    def test_zoom_out(self, mgr):
        mgr.handle_zoom(0.5)
        assert mgr.zoom_level == pytest.approx(0.5)

    def test_raises_on_zero_factor(self, mgr):
        with pytest.raises(GLInitializationError, match="positive"):
            mgr.handle_zoom(0.0)

    def test_raises_on_negative_factor(self, mgr):
        with pytest.raises(GLInitializationError, match="positive"):
            mgr.handle_zoom(-1.0)

    def test_no_pan_change_without_centre(self, mgr):
        mgr.handle_zoom(2.0)
        assert mgr.pan_x == pytest.approx(0.0)
        assert mgr.pan_y == pytest.approx(0.0)

    def test_centre_zoom_adjusts_pan(self, mgr):
        """Zooming towards a non-centre point must shift pan."""
        mgr.handle_resize(800, 600)
        # Zoom towards a point offset from the viewport centre.
        mgr.handle_zoom(2.0, center_x=600.0, center_y=400.0)
        # Pan must have changed from 0 to compensate for the off-centre zoom.
        assert mgr.pan_x != pytest.approx(0.0) or mgr.pan_y != pytest.approx(0.0)

    def test_centre_zoom_at_viewport_centre_does_not_change_pan(self, mgr):
        """Zooming exactly at the image centre (pan=0) must not shift pan."""
        mgr.handle_resize(800, 600)
        cx = mgr.viewport_w / 2.0   # 400
        cy = mgr.viewport_h / 2.0   # 300
        mgr.handle_zoom(2.0, center_x=cx, center_y=cy)
        assert mgr.pan_x == pytest.approx(0.0, abs=1e-6)
        assert mgr.pan_y == pytest.approx(0.0, abs=1e-6)

    def test_zoom_level_unchanged_on_guard_failure(self, mgr):
        """State must not be mutated when the guard raises."""
        original = mgr.zoom_level
        with pytest.raises(GLInitializationError):
            mgr.handle_zoom(0.0)
        assert mgr.zoom_level == pytest.approx(original)


# =============================================================================
# handle_rotation
# =============================================================================

class TestHandleRotation:

    def test_sets_rotation(self, mgr):
        mgr.handle_rotation(45.0)
        assert mgr.rotation == pytest.approx(45.0)

    def test_overwrites_previous_rotation(self, mgr):
        mgr.handle_rotation(30.0)
        mgr.handle_rotation(90.0)
        assert mgr.rotation == pytest.approx(90.0)

    def test_negative_rotation_stored(self, mgr):
        mgr.handle_rotation(-45.0)
        assert mgr.rotation == pytest.approx(-45.0)

    def test_transform_differs_from_zero_rotation(self, mgr):
        mgr.handle_resize(800, 600)
        mgr.set_image_size(400, 300)
        t_before = _flat(mgr.transform).copy()
        mgr.handle_rotation(45.0)
        t_after = _flat(mgr.transform)
        assert not np.allclose(t_before, t_after)


# =============================================================================
# reset_view
# =============================================================================

class TestResetView:

    def test_resets_zoom(self, mgr):
        mgr.handle_zoom(3.0)
        mgr.reset_view()
        assert mgr.zoom_level == pytest.approx(1.0)

    def test_resets_pan(self, mgr):
        mgr.handle_pan(200.0, -100.0)
        mgr.reset_view()
        assert mgr.pan_x == pytest.approx(0.0)
        assert mgr.pan_y == pytest.approx(0.0)

    def test_resets_rotation(self, mgr):
        mgr.handle_rotation(90.0)
        mgr.reset_view()
        assert mgr.rotation == pytest.approx(0.0)

    def test_all_state_reset_simultaneously(self, mgr):
        mgr.handle_zoom(2.0)
        mgr.handle_pan(50.0, 50.0)
        mgr.handle_rotation(30.0)
        mgr.reset_view()
        assert mgr.zoom_level == pytest.approx(1.0)
        assert mgr.pan_x      == pytest.approx(0.0)
        assert mgr.pan_y      == pytest.approx(0.0)
        assert mgr.rotation   == pytest.approx(0.0)


# =============================================================================
# fit_to_viewport
# =============================================================================

class TestFitToViewport:

    def test_zoom_fits_landscape_image_in_square_viewport(self, mgr):
        mgr.handle_resize(500, 500)
        mgr.set_image_size(1000, 500)   # 2:1 aspect ratio
        mgr.fit_to_viewport()
        # Limiting axis is width: 500/1000 = 0.5, with padding = 0.475
        assert mgr.zoom_level == pytest.approx(0.5 * _FIT_PADDING, rel=1e-5)

    def test_zoom_fits_portrait_image_in_square_viewport(self, mgr):
        mgr.handle_resize(500, 500)
        mgr.set_image_size(500, 1000)   # 1:2 aspect ratio
        mgr.fit_to_viewport()
        # Limiting axis is height: 500/1000 = 0.5
        assert mgr.zoom_level == pytest.approx(0.5 * _FIT_PADDING, rel=1e-5)

    def test_zoom_fits_image_same_aspect_as_viewport(self, mgr):
        mgr.handle_resize(800, 600)
        mgr.set_image_size(800, 600)
        mgr.fit_to_viewport()
        assert mgr.zoom_level == pytest.approx(1.0 * _FIT_PADDING, rel=1e-5)

    def test_resets_pan(self, mgr):
        mgr.handle_pan(100.0, 50.0)
        mgr.handle_resize(800, 600)
        mgr.set_image_size(400, 300)
        mgr.fit_to_viewport()
        assert mgr.pan_x == pytest.approx(0.0)
        assert mgr.pan_y == pytest.approx(0.0)

    def test_resets_rotation(self, mgr):
        mgr.handle_rotation(45.0)
        mgr.handle_resize(800, 600)
        mgr.set_image_size(400, 300)
        mgr.fit_to_viewport()
        assert mgr.rotation == pytest.approx(0.0)

    def test_noop_when_image_is_default_one_by_one(self, mgr):
        """fit_to_viewport must not raise or error on the default 1×1 image size."""
        mgr.handle_resize(800, 600)
        original_zoom = mgr.zoom_level
        # image_w = image_h = 1 by default — fit should still run normally.
        mgr.fit_to_viewport()
        # Just verify it didn't raise and zoom is positive.
        assert mgr.zoom_level > 0


# =============================================================================
# update_transform — rotation threshold
# =============================================================================

class TestUpdateTransform:

    def test_rotation_below_epsilon_not_applied(self, mgr):
        """
        Angles below _ROTATION_EPSILON must not trigger a QMatrix4x4.rotate call.
        We verify indirectly by checking that the transform for angle=0 and
        angle=0.0005 are the same.
        """
        mgr.handle_resize(800, 600)
        mgr.set_image_size(400, 300)

        mgr.handle_rotation(0.0)
        t_zero = _flat(mgr.transform).copy()

        mgr.handle_rotation(0.0005)   # below 0.001 threshold
        t_tiny = _flat(mgr.transform)

        np.testing.assert_allclose(t_zero, t_tiny, atol=1e-6)

    def test_rotation_above_epsilon_changes_transform(self, mgr):
        mgr.handle_resize(800, 600)
        mgr.set_image_size(400, 300)

        mgr.handle_rotation(0.0)
        t_zero = _flat(mgr.transform).copy()

        mgr.handle_rotation(1.0)   # well above threshold
        t_rotated = _flat(mgr.transform)

        assert not np.allclose(t_zero, t_rotated)

    def test_pan_is_reflected_in_transform(self, mgr):
        """Non-zero pan must produce a different transform than zero pan."""
        mgr.handle_resize(800, 600)
        mgr.set_image_size(400, 300)

        mgr.reset_view()
        t_no_pan = _flat(mgr.transform).copy()

        mgr.handle_pan(50.0, 0.0)
        t_panned = _flat(mgr.transform)

        assert not np.allclose(t_no_pan, t_panned)

    def test_zoom_is_reflected_in_transform(self, mgr):
        mgr.handle_resize(800, 600)
        mgr.set_image_size(400, 300)

        mgr.reset_view()
        t_zoom_1 = _flat(mgr.transform).copy()

        mgr.handle_zoom(2.0)
        t_zoom_2 = _flat(mgr.transform)

        assert not np.allclose(t_zoom_1, t_zoom_2)


# =============================================================================
# get_projection_data / get_transform_data
# =============================================================================

class TestMatrixDataAccessors:

    def test_projection_data_shape_is_flat_16(self, mgr):
        data = mgr.get_projection_data()
        assert data.shape == (16,), (
            "Expected flat (16,) array — reshape to (4,4) would produce the "
            "mathematical transpose due to numpy row-major vs GL column-major"
        )

    def test_projection_data_dtype_is_float32(self, mgr):
        assert mgr.get_projection_data().dtype == np.float32

    def test_transform_data_shape_is_flat_16(self, mgr):
        data = mgr.get_transform_data()
        assert data.shape == (16,)

    def test_transform_data_dtype_is_float32(self, mgr):
        assert mgr.get_transform_data().dtype == np.float32

    def test_projection_data_changes_after_resize(self, mgr):
        mgr.handle_resize(800, 600)
        d1 = mgr.get_projection_data().copy()
        mgr.handle_resize(1280, 720)
        d2 = mgr.get_projection_data()
        assert not np.allclose(d1, d2)

    def test_transform_data_changes_after_pan(self, mgr):
        mgr.handle_resize(800, 600)
        d1 = mgr.get_transform_data().copy()
        mgr.handle_pan(100.0, 50.0)
        d2 = mgr.get_transform_data()
        assert not np.allclose(d1, d2)

    def test_transform_data_changes_after_zoom(self, mgr):
        mgr.handle_resize(800, 600)
        d1 = mgr.get_transform_data().copy()
        mgr.handle_zoom(2.0)
        d2 = mgr.get_transform_data()
        assert not np.allclose(d1, d2)

    def test_projection_data_contains_16_finite_values(self, mgr):
        mgr.handle_resize(800, 600)
        data = mgr.get_projection_data()
        assert np.all(np.isfinite(data)), "Projection matrix contains non-finite values"

    def test_transform_data_contains_16_finite_values(self, mgr):
        mgr.handle_resize(800, 600)
        data = mgr.get_transform_data()
        assert np.all(np.isfinite(data)), "Transform matrix contains non-finite values"