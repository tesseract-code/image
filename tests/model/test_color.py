"""
Test suite for ColorOptimizer class.
"""
import pytest
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap


class TestColorOptimizer:
    """Test suite for ColorOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create ColorOptimizer instance."""
        from cross_platform.qt6_utils.image.model.cmap import ColorOptimizer
        return ColorOptimizer()

    # Basic functionality tests

    def test_get_contrasting_color_returns_valid_rgb(self, optimizer):
        """Test that returned color is valid RGB tuple."""
        color = optimizer.get_contrasting_color('viridis')

        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 1 for c in color)

    def test_accepts_string_colormap_name(self, optimizer):
        """Test string colormap names work."""
        color = optimizer.get_contrasting_color('plasma')
        assert color is not None

    def test_accepts_colormap_object(self, optimizer):
        """Test Colormap objects work."""
        cmap = colormaps['coolwarm']
        color = optimizer.get_contrasting_color(cmap)
        assert color is not None

    def test_sample_points_parameter(self, optimizer):
        """Test different sample point values."""
        color1 = optimizer.get_contrasting_color('viridis', sample_points=10)
        color2 = optimizer.get_contrasting_color('viridis', sample_points=100)

        # Should return valid colors regardless
        assert color1 is not None
        assert color2 is not None

    # Specific colormap tests

    def test_dark_colormap_returns_light_color(self, optimizer):
        """Dark colormaps should return light contrasting colors."""
        color = optimizer.get_contrasting_color('magma')
        luminance = optimizer._relative_luminance(color)

        # Should be relatively bright
        assert luminance > 0.4

    def test_light_colormap_returns_dark_color(self, optimizer):
        """Light colormaps should return dark contrasting colors."""
        # Create a light colormap
        cmap = LinearSegmentedColormap.from_list(
            'light', [(0.8, 0.8, 0.8), (1.0, 1.0, 1.0)]
        )
        color = optimizer.get_contrasting_color(cmap)
        luminance = optimizer._relative_luminance(color)

        # Should be relatively dark
        assert luminance < 0.3

    def test_diverging_colormap(self, optimizer):
        """Test diverging colormaps like coolwarm."""
        color = optimizer.get_contrasting_color('coolwarm')

        # Should return a valid contrasting color
        assert color in optimizer.CANDIDATES

    def test_sequential_colormap(self, optimizer):
        """Test sequential colormaps like viridis."""
        color = optimizer.get_contrasting_color('viridis')
        assert color is not None

    # Qt integration test

    def test_qt_color_conversion(self, optimizer):
        """Test QColor conversion (if PyQt5 available)."""
        pytest.importorskip('PyQt6')
        from PyQt6.QtGui import QColor

        qcolor = optimizer.get_contrasting_color_qt('viridis')

        assert isinstance(qcolor, QColor)
        assert qcolor.isValid()
        assert 0 <= qcolor.red() <= 255
        assert 0 <= qcolor.green() <= 255
        assert 0 <= qcolor.blue() <= 255

    # Contrast ratio tests

    def test_contrast_ratio_maximum(self, optimizer):
        """Test maximum contrast (black vs white)."""
        ratio = optimizer._contrast_ratio((0, 0, 0), (1, 1, 1))
        assert abs(ratio - 21.0) < 0.1  # Maximum contrast is 21:1

    def test_contrast_ratio_minimum(self, optimizer):
        """Test minimum contrast (identical colors)."""
        ratio = optimizer._contrast_ratio((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert abs(ratio - 1.0) < 0.01  # Same color is 1:1

    def test_contrast_ratio_symmetric(self, optimizer):
        """Test that contrast ratio is symmetric."""
        color1 = (0.2, 0.3, 0.4)
        color2 = (0.7, 0.8, 0.9)

        ratio1 = optimizer._contrast_ratio(color1, color2)
        ratio2 = optimizer._contrast_ratio(color2, color1)

        assert abs(ratio1 - ratio2) < 0.001

    # Relative luminance tests

    def test_relative_luminance_black(self, optimizer):
        """Test luminance of black is 0."""
        lum = optimizer._relative_luminance((0, 0, 0))
        assert abs(lum) < 0.001

    def test_relative_luminance_white(self, optimizer):
        """Test luminance of white is 1."""
        lum = optimizer._relative_luminance((1, 1, 1))
        assert abs(lum - 1.0) < 0.001

    def test_relative_luminance_ordering(self, optimizer):
        """Test that luminance increases with brightness."""
        dark = optimizer._relative_luminance((0.2, 0.2, 0.2))
        mid = optimizer._relative_luminance((0.5, 0.5, 0.5))
        light = optimizer._relative_luminance((0.8, 0.8, 0.8))

        assert dark < mid < light

    # Analysis tests

    def test_analyze_colormap_structure(self, optimizer):
        """Test that analyze_colormap returns expected structure."""
        analysis = optimizer.analyze_colormap('viridis')

        assert 'colormap' in analysis
        assert 'best_color' in analysis
        assert 'best_min_contrast' in analysis
        assert 'all_candidates' in analysis

        assert isinstance(analysis['all_candidates'], dict)
        assert len(analysis['all_candidates']) == len(optimizer.CANDIDATES)

    def test_analyze_colormap_best_color_valid(self, optimizer):
        """Test that best color is from candidates."""
        analysis = optimizer.analyze_colormap('plasma')

        assert analysis['best_color'] in optimizer.CANDIDATES
        assert analysis['best_min_contrast'] > 0

    def test_analyze_colormap_candidate_stats(self, optimizer):
        """Test that candidate stats are valid."""
        analysis = optimizer.analyze_colormap('coolwarm')

        for candidate, stats in analysis['all_candidates'].items():
            assert 'min' in stats
            assert 'max' in stats
            assert 'mean' in stats
            assert 'median' in stats

            # Logical checks
            assert stats['min'] <= stats['mean'] <= stats['max']
            assert stats['min'] <= stats['median'] <= stats['max']
            assert stats['min'] >= 1.0  # Minimum possible contrast

    # Edge cases

    def test_custom_colormap(self, optimizer):
        """Test with custom colormap."""
        cmap = LinearSegmentedColormap.from_list(
            'custom', ['red', 'blue']
        )
        color = optimizer.get_contrasting_color(cmap)
        assert color is not None

    def test_grayscale_colormap(self, optimizer):
        """Test with grayscale colormap."""
        color = optimizer.get_contrasting_color('gray')

        # For grayscale, algorithm picks color with best min contrast
        # (e.g., blue has ~2.4:1 min vs white's 1:1 min)
        # This is correct - consistent contrast beats extreme contrast at one end
        contrast_with_mid_gray = optimizer._contrast_ratio(
            color, (0.5, 0.5, 0.5)
        )
        assert contrast_with_mid_gray > 2.0  # Reasonable consistent contrast

    def test_grayscale_min_contrast_strategy(self, optimizer):
        """Verify that grayscale uses min-contrast strategy correctly."""
        analysis = optimizer.analyze_colormap('gray')

        # The chosen color should have better minimum contrast than white/black
        best_min = analysis['best_min_contrast']
        white_min = analysis['all_candidates'][(1.0, 1.0, 1.0)]['min']
        black_min = analysis['all_candidates'][(0.0, 0.0, 0.0)]['min']

        # Best color should have higher min contrast than pure white or black
        assert best_min > white_min
        assert best_min > black_min
    # Consistency tests

    def test_deterministic_results(self, optimizer):
        """Test that same inputs give same outputs."""
        color1 = optimizer.get_contrasting_color('viridis', sample_points=50)
        color2 = optimizer.get_contrasting_color('viridis', sample_points=50)

        assert color1 == color2

    def test_common_colormaps(self, optimizer):
        """Test with commonly used colormaps."""
        common_cmaps = ['viridis', 'plasma', 'inferno', 'magma',
                        'cividis', 'twilight', 'coolwarm', 'jet']

        for cmap_name in common_cmaps:
            color = optimizer.get_contrasting_color(cmap_name)
            assert color is not None
            assert color in optimizer.CANDIDATES


# Pytest configuration for running tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
