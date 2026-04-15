import unittest
import numpy as np

try:
    import skimage  # noqa: F401
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from core.models.spx import SPX_Tester2D, SPX_SLIC2D, SPX_Felzenszwalb2D


def _gray_image(h=32, w=32):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _rgb_image(h=32, w=32):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestSPXTester2D(unittest.TestCase):

    def setUp(self):
        self.model = SPX_Tester2D()

    def test_output_shape_matches_2d_input(self):
        img = _gray_image(20, 30)
        labels = self.model.forward(img=img)
        self.assertEqual(labels.shape, (20, 30))

    def test_output_shape_matches_3d_input(self):
        img = _rgb_image(20, 30)
        labels = self.model.forward(img=img)
        self.assertEqual(labels.shape, (20, 30))

    def test_all_labels_are_positive(self):
        img = _gray_image()
        labels = self.model.forward(img=img)
        self.assertTrue((labels >= 1).all())

    def test_respects_gh_gw_kwargs(self):
        """After the bug fix, gh=3/gw=4 must produce exactly 3*4=12 unique labels."""
        img = _gray_image(60, 80)
        labels = self.model.forward(img=img, gh=3, gw=4)
        unique = np.unique(labels)
        self.assertEqual(len(unique), 3 * 4)

    def test_default_gh_gw_produces_81_labels(self):
        img = _gray_image(90, 90)
        labels = self.model.forward(img=img)
        self.assertEqual(len(np.unique(labels)), 9 * 9)

    def test_integer_conversion_of_gh_gw(self):
        """Float gh/gw values are converted without error."""
        img = _gray_image()
        labels = self.model.forward(img=img, gh=3.0, gw=3.0)
        self.assertEqual(labels.shape, img.shape)

    def test_output_dtype_is_int32(self):
        img = _gray_image()
        labels = self.model.forward(img=img)
        self.assertEqual(labels.dtype, np.int32)

    def test_vectorized_grid_cells_are_uniform(self):
        """Within each grid cell every pixel must share the same label.

        Uses an image size that divides evenly (H=60/gh=3, W=80/gw=4) so each
        cell is exactly 20×20 pixels — any intra-cell label change would be a
        vectorisation bug.
        """
        H, W, gh, gw = 60, 80, 3, 4
        cell_h, cell_w = H // gh, W // gw
        img = _gray_image(H, W)
        labels = self.model.forward(img=img, gh=gh, gw=gw)
        for row in range(gh):
            for col in range(gw):
                cell = labels[row * cell_h:(row + 1) * cell_h,
                               col * cell_w:(col + 1) * cell_w]
                unique = np.unique(cell)
                self.assertEqual(len(unique), 1,
                                 f"Cell ({row},{col}) has {len(unique)} labels: {unique} — "
                                 "vectorized grid contains spurious intra-cell boundaries")

    def test_large_image_no_loop_overhead(self):
        """Vectorized path must handle a 512×512 image without error or OOM."""
        img = _gray_image(512, 512)
        labels = self.model.forward(img=img, gh=16, gw=16)
        self.assertEqual(labels.shape, (512, 512))
        self.assertEqual(len(np.unique(labels)), 16 * 16)


@unittest.skipUnless(SKIMAGE_AVAILABLE, "scikit-image not installed")
class TestSPXSLIC2D(unittest.TestCase):

    def setUp(self):
        self.model = SPX_SLIC2D()

    def test_returns_integer_label_map_of_correct_shape(self):
        img = _gray_image(64, 64)
        labels = self.model.forward(img=img, n_segments=10)
        self.assertEqual(labels.shape, (64, 64))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))

    def test_raises_on_none_img(self):
        with self.assertRaises(ValueError):
            self.model.forward(img=None)

    def test_passes_kwargs_to_slic(self):
        img = _gray_image(64, 64)
        labels_few = self.model.forward(img=img, n_segments=4)
        labels_many = self.model.forward(img=img, n_segments=50)
        # More requested segments → at least as many unique labels
        self.assertLessEqual(len(np.unique(labels_few)), len(np.unique(labels_many)))

    def test_img_popped_from_kwargs(self):
        """Forward must not pass 'img' down to slic (it pops it from kwargs)."""
        img = _gray_image()
        # If img were forwarded, skimage would receive an unexpected kwarg and raise;
        # the absence of an error confirms it was popped correctly.
        self.model.forward(img=img)


@unittest.skipUnless(SKIMAGE_AVAILABLE, "scikit-image not installed")
class TestSPXFelzenszwalb2D(unittest.TestCase):

    def setUp(self):
        self.model = SPX_Felzenszwalb2D()

    def test_returns_integer_label_map_of_correct_shape(self):
        img = _gray_image(64, 64)
        labels = self.model.forward(img=img)
        self.assertEqual(labels.shape, (64, 64))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))

    def test_raises_on_none_img(self):
        with self.assertRaises(ValueError):
            self.model.forward(img=None)

    def test_constant_image_does_not_raise(self):
        """When img_min == img_max, normalization must be skipped (no divide-by-zero)."""
        img = np.full((32, 32), 128, dtype=np.uint8)
        labels = self.model.forward(img=img)
        self.assertEqual(labels.shape, (32, 32))

    def test_normalizes_to_float(self):
        """Internally the image must be cast to float32 before forwarding."""
        img = np.array([[0, 128], [64, 255]], dtype=np.uint8)
        # If normalization works correctly, the result should not raise or
        # produce negative labels.
        labels = self.model.forward(img=img)
        self.assertTrue((labels >= 0).all())


if __name__ == '__main__':
    unittest.main()
