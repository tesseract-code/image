"""
Image data model interface with caching for hot paths.
"""

from typing import Optional, Union, Any

import numpy as np
import numpy.typing as npt

from image.model.utils import get_value_at_position, get_roi
from image.pipeline.stats import FrameStats, get_frame_stats
from image.utils.types import is_image


class ImageDataModel:
    """
    Centralized image data model with caching for performance-critical paths.

    This class serves as the sole source of image data and provides efficient
    access patterns for frequently called methods like get_resolution() and get_data().
    """

    def __init__(self) -> None:
        """Initialize an empty image data model."""
        self._data: Optional[npt.NDArray[Any]] = None
        self._metadata: Optional[FrameStats] = None

        # Cache for hot paths - invalidated on set_data()
        self._cached_readonly_view: Optional[npt.NDArray[Any]] = None

    def has_data(self) -> bool:
        """
        Check if the model contains image data.

        Returns:
            bool: True if image data is present, False otherwise.
        """
        return self._data is not None

    def set_data(self, data: npt.NDArray[Any], copy: bool = True) -> None:
        """
        Set image data. This is the ONLY method that performs data copying.

        Args:
            data: Image data as numpy array. Expected shapes:
                  - (height, width) for grayscale
                  - (height, width, channels) for color
                  Supports any numpy dtype (uint8, uint16, float32, etc.)
            copy: If True (default), creates a copy of the data.
                  If False, takes ownership of the array (use with caution).

        Raises:
            ValueError: If data is not a 2D or 3D array.

        Note:
            This method invalidates all cached values.
            The dtype of the input is preserved - no conversion is performed.
            WARNING: set_data(copy=False) transfers ownership - caller must not modify.
        """
        # Validate array dimensions
        data_array = np.asarray(data)
        if not is_image(data):
            raise ValueError(
                f"Image data must be a standard image!"
            )

        if copy:
            self._data = data_array.copy()
        else:
            # Take ownership without copying - caller must not modify original
            # Ensure we have a base array, not a view
            if not data_array.flags.owndata:
                # If it's a view, we must copy to ensure ownership
                self._data = data_array.copy()
            else:
                self._data = data_array

        self._metadata = get_frame_stats(self._data)

        # Invalidate caches
        self._invalidate_cache()

    def get_data(self, copy: bool = False) -> Optional[npt.NDArray[Any]]:
        """
        Get image data. Returns a READ-ONLY view by default for performance.

        Args:
            copy: If True, returns a writeable copy. If False (default), returns
                  a read-only view. Use copy=True when you need to modify data.

        Returns:
            Image data array or None if no data is present.
            Views are READ-ONLY to enforce set_data() as sole authority for changes.

        Note:
            For hot paths, use copy=False (default) to avoid allocation overhead.
            Attempts to modify the view will raise ValueError.
            The read-only view is cached for performance.
        """
        if self._data is None:
            return None

        if copy:
            return self._data.copy()
        else:
            # Return cached READ-ONLY view for hot path performance
            if self._cached_readonly_view is None:
                view = self._data.view()
                view.flags.writeable = False
                self._cached_readonly_view = view
            return self._cached_readonly_view

    def get_metadata(self):
        return self._metadata

    def get_view(self) -> Optional[npt.NDArray[Any]]:
        """
        Get a READ-ONLY view of the image data (alias for get_data(copy=False)).

        Returns:
            A read-only view of the image data, or None if no data is present.

        Note:
            This enforces set_data() as the sole authority for model changes.
            Attempting to modify the view will raise ValueError.
        """
        return self.get_data(copy=False)

    def get_copy(self) -> Optional[npt.NDArray[Any]]:
        """
        Get a copy of the image data (alias for get_data(copy=True)).

        Returns:
            An independent copy of the image data, or None if no data is present.
        """
        return self.get_data(copy=True)

    def get_shape(self) -> Optional[tuple[int, ...]]:
        """
        Get the shape of the image data (cached for performance).

        Returns:
            Tuple of (height, width) or (height, width, channels), or None.

        Note:
            This method is cached and very fast for repeated calls.
        """
        if self._data is None:
            return None

        return self._metadata.shape

    def get_resolution(self) -> Optional[tuple[int, int]]:
        """
        Get the resolution (width, height) of the image (cached for performance).

        Returns:
            Tuple of (width, height), or None if no data is present.

        Note:
            This is a hot path method and is heavily cached.
            Returns width first (standard convention), even though numpy uses
            (height, width) ordering.
        """
        if self._data is None:
            return None

        height, width = self._metadata.shape[:2]

        return width, height

    def get_dtype(self) -> Optional[np.dtype]:
        """
        Get the data type of the image data.

        Returns:
            The numpy dtype, or None if no data is present.
        """
        return self._data.dtype if self.has_data() else None

    def get_channels(self) -> Optional[int]:
        """
        Get the number of channels in the image.

        Returns:
            Number of channels (1 for grayscale, 3 for RGB, 4 for RGBA, etc.),
            or None if no data is present.
        """
        if self._data is None:
            return None

        return self._metadata.shape[2] if self._data.ndim == 3 else 1

    def get_value_at(
            self,
            x: Union[int, float],
            y: Union[int, float],
            flip_x: bool = False,
            flip_y: bool = False
    ) -> Optional[Union[np.generic, npt.NDArray[Any]]]:
        """
        Get pixel value at specified position.

        Args:
            x: X coordinate (column)
            y: Y coordinate (row)
            flip_x: If True, flip X axis for scene/graphics coordinates
            flip_y: If True, flip Y axis for scene/graphics coordinates

        Returns:
            Pixel value or None if out of bounds or no data.
            Return type matches the image's dtype.

        Example:
            >>> value = model.get_value_at(100, 200)  # Get pixel at (100, 200)
        """
        if self._data is None:
            return None

        return get_value_at_position(self._data, x, y, flip_x, flip_y)

    def get_region(
            self,
            x: int,
            y: int,
            width: int,
            height: int,
            copy: bool = True
    ) -> Optional[npt.NDArray[Any]]:
        """
        Extract a rectangular region of interest (ROI).

        Args:
            x: Left edge (column) of region
            y: Top edge (row) of region
            width: Region width in pixels
            height: Region height in pixels
            copy: If True (default), return a copy. If False, return read-only view.

        Returns:
            Region array or None if invalid or no data.
            Return type matches the image's dtype.

        Example:
            >>> roi = model.get_region(100, 100, 50, 50)  # 50x50 region
        """
        if self._data is None:
            return None

        return get_roi(self._data, x, y, width, height, copy)

    def clear(self) -> None:
        """Clear the image data and invalidate all caches."""
        self._data = None
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate all cached values."""
        self._cached_resolution = None
        self._cached_shape = None
        self._cached_readonly_view = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        if not self.has_data():
            return "ImageDataModel(empty)"

        shape = self.get_shape()
        resolution = self.get_resolution()
        channels = self.get_channels()

        return (f"ImageDataModel(resolution={resolution}, "
                f"shape={shape}, channels={channels}, "
                f"dtype={self.get_dtype()})")
