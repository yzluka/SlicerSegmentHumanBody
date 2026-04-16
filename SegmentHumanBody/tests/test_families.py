import unittest
import numpy as np

from core.modelFamilies import BaseModelFamily, SAMFamily, SPXModelFamily, AutoModelFamily
from core.modelRegistry import ModelRegistry


class _FakeModel:
    """Deterministic superpixel model used as a stand-in for real models."""

    PARAM_HINT = 'fake=1'

    def forward(self, **kwargs):
        img = kwargs.get('img')
        H, W = img.shape[:2] if img.ndim >= 2 else (img.shape[0], 1)
        labels = np.zeros((H, W), dtype=np.int32)
        # Two regions: left half = label 1, right half = label 2
        labels[:, : W // 2] = 1
        labels[:, W // 2 :] = 2
        return labels


class TestBaseModelFamily(unittest.TestCase):

    def setUp(self):
        ModelRegistry.model_cache.clear()

    def tearDown(self):
        ModelRegistry.model_cache.clear()

    def test_confirm_model_no_variant_is_noop(self):
        fam = BaseModelFamily(variant=None)
        fam.confirm_model()          # must not raise
        self.assertIsNone(fam.model)

    def test_confirm_model_sets_model_from_registry(self):
        fam = BaseModelFamily(variant='SPX_Tester2D')
        fam.confirm_model()
        self.assertIsNotNone(fam.model)


class TestSAMFamily(unittest.TestCase):
    """SAMFamily methods are intentional stubs (not yet implemented).
    Tests confirm they exist, accept **kwargs, and do not raise."""

    def setUp(self):
        self.fam = SAMFamily(variant='SAM-VIT-H')

    def test_on_enter_interactive_does_not_raise(self):
        self.fam.on_enter_interactive()
        self.fam.on_enter_interactive(extra='ignored')

    def test_on_stop_interactive_does_not_raise(self):
        self.fam.on_stop_interactive()
        self.fam.on_stop_interactive(extra='ignored')

    def test_get_requested_mask_does_not_raise(self):
        self.fam.get_requested_mask()
        self.fam.get_requested_mask(mask_index=0)

    def test_onrender_does_not_raise_and_returns_none(self):
        result = self.fam.onRender(img=np.zeros((10, 10)), pos_points=[], neg_points=[])
        self.assertIsNone(result)


class TestSPXModelFamily(unittest.TestCase):

    # ---- _get_model_key ----

    def test_get_model_key_slic(self):
        fam = SPXModelFamily(variant='SLIC-2D')
        self.assertEqual(fam._get_model_key(), 'SPX_SLIC2D')

    def test_get_model_key_felzenszwalb(self):
        fam = SPXModelFamily(variant='Felzenszwalb-2D')
        self.assertEqual(fam._get_model_key(), 'SPX_Felzenszwalb2D')

    def test_get_model_key_naive_grid(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        self.assertEqual(fam._get_model_key(), 'SPX_Tester2D')

    def test_get_model_key_no_variant_raises(self):
        fam = SPXModelFamily(variant=None)
        with self.assertRaises(ValueError):
            fam._get_model_key()

    def test_get_model_key_unknown_variant_raises(self):
        fam = SPXModelFamily(variant='UnknownVariant')
        with self.assertRaises(ValueError):
            fam._get_model_key()

    # ---- onRender ----

    def _make_family_with_fake_model(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _FakeModel()
        return fam

    def test_onrender_returns_none_with_no_pos_points(self):
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 10), dtype=np.uint8)
        result = fam.onRender(img=img, pos_points=[], neg_points=[])
        self.assertIsNone(result)

    def test_onrender_returns_uint8_mask(self):
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 20), dtype=np.uint8)
        # Point in left half → selects label 1
        result = fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[])
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (10, 20))

    def test_onrender_selects_correct_region(self):
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 20), dtype=np.uint8)
        # Left half: x in [0,9]; _FakeModel labels left half = 1
        result = fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[])
        # Left half should be 1, right half 0
        np.testing.assert_array_equal(result[:, :10], 1)
        np.testing.assert_array_equal(result[:, 10:], 0)

    def test_onrender_returns_none_when_no_model(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        # model is None (confirm_model not called)
        img = np.zeros((10, 10), dtype=np.uint8)
        result = fam.onRender(img=img, pos_points=[[0, 0]], neg_points=[])
        self.assertIsNone(result)

    def test_onrender_out_of_bounds_point_is_ignored(self):
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 10), dtype=np.uint8)
        # x=999, y=999 → far outside (10,10) label array
        result = fam.onRender(img=img, pos_points=[[999, 999]], neg_points=[])
        # No valid label selected → None
        self.assertIsNone(result)

    def test_onrender_multiple_pos_points_union_regions(self):
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 20), dtype=np.uint8)
        # One point in each half → both labels selected → full mask
        result = fam.onRender(img=img, pos_points=[[3, 5], [15, 5]], neg_points=[])
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, 1)

    def test_onrender_neg_point_subtracts_region(self):
        """Neg point on right half removes that label from the result."""
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 20), dtype=np.uint8)
        # pos selects both halves; neg removes right half (label 2)
        result = fam.onRender(img=img, pos_points=[[3, 5], [15, 5]], neg_points=[[15, 5]])
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result[:, :10], 1)   # left half kept
        np.testing.assert_array_equal(result[:, 10:], 0)   # right half removed

    def test_onrender_neg_cancels_all_pos_returns_none(self):
        """If neg removes every selected label, result is None."""
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 20), dtype=np.uint8)
        # pos selects left half only; neg also on left half → nothing left
        result = fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[[3, 5]])
        self.assertIsNone(result)

    # ---- base_mask (interactive additive / subtractive mode) ----

    def test_onrender_with_base_mask_neg_erases_existing_region(self):
        """Neg-only (no pos): neg point erases its SPX region from the base mask."""
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 20), dtype=np.uint8)
        # Base: entire image painted.
        base = np.ones((10, 20), dtype=np.uint8)
        # Neg point on right half (label 2) → right half cleared, left kept.
        result = fam.onRender(img=img, pos_points=[], neg_points=[[15, 5]], base_mask=base)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result[:, :10], 1)
        np.testing.assert_array_equal(result[:, 10:], 0)

    def test_onrender_with_base_mask_pos_adds_to_existing(self):
        """Pos point adds its SPX region on top of the base mask."""
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 20), dtype=np.uint8)
        # Base: only right half painted.
        base = np.zeros((10, 20), dtype=np.uint8)
        base[:, 10:] = 1
        # Pos on left half → left half added, right half kept from base.
        result = fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[], base_mask=base)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, 1)

    def test_onrender_with_base_mask_neg_wins_over_pos(self):
        """When pos and neg point to the same SPX region, neg wins (region erased)."""
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 20), dtype=np.uint8)
        base = np.ones((10, 20), dtype=np.uint8)
        # Both pos and neg on left half → left erased, right kept from base.
        result = fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[[3, 5]], base_mask=base)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result[:, :10], 0)
        np.testing.assert_array_equal(result[:, 10:], 1)

    def test_onrender_with_base_mask_preserves_base_when_no_matching_prompts(self):
        """Out-of-bounds prompts → base mask returned unchanged."""
        fam = self._make_family_with_fake_model()
        img = np.zeros((10, 20), dtype=np.uint8)
        base = np.ones((10, 20), dtype=np.uint8)
        result = fam.onRender(img=img, pos_points=[[999, 999]], neg_points=[], base_mask=base)
        # 999,999 is out of bounds → pos_only_labels is empty → base unchanged
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, base)

    def test_onrender_base_mask_not_included_in_spx_cache_key(self):
        """Different base_mask values must not cause extra model.forward() calls."""
        call_count = [0]
        original_forward = _FakeModel.forward

        class _CountingModel(_FakeModel):
            def forward(self, **kwargs):
                call_count[0] += 1
                return original_forward(self, **kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()
        img = np.zeros((10, 20), dtype=np.uint8)
        base_a = np.zeros((10, 20), dtype=np.uint8)
        base_b = np.ones((10, 20), dtype=np.uint8)

        fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[], base_mask=base_a)
        fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[], base_mask=base_b)

        self.assertEqual(call_count[0], 1,
                         "Different base_mask values must not bust the SPX label cache")

    # ---- SPX label cache ----

    def test_cache_hit_skips_forward(self):
        """Same img buffer + same kwargs → model.forward called exactly once."""
        call_count = [0]
        original_forward = _FakeModel.forward

        class _CountingModel(_FakeModel):
            def forward(self, **kwargs):
                call_count[0] += 1
                return original_forward(self, **kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()

        img = np.zeros((10, 20), dtype=np.uint8)
        fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[])
        fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[])
        fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[])

        self.assertEqual(call_count[0], 1, "forward() should be called once for repeated renders on the same slice")

    def test_cache_miss_on_different_image(self):
        """Different image buffer → forward called again."""
        call_count = [0]
        original_forward = _FakeModel.forward

        class _CountingModel(_FakeModel):
            def forward(self, **kwargs):
                call_count[0] += 1
                return original_forward(self, **kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()

        img1 = np.zeros((10, 20), dtype=np.uint8)
        img2 = np.ones((10, 20), dtype=np.uint8)  # different buffer

        fam.onRender(img=img1, pos_points=[[3, 5]], neg_points=[])
        fam.onRender(img=img2, pos_points=[[3, 5]], neg_points=[])

        self.assertEqual(call_count[0], 2)

    def test_cache_miss_on_different_kwargs(self):
        """Same image but different user params → forward called again."""
        call_count = [0]
        original_forward = _FakeModel.forward

        class _CountingModel(_FakeModel):
            def forward(self, **kwargs):
                call_count[0] += 1
                return original_forward(self, **kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()

        img = np.zeros((10, 20), dtype=np.uint8)
        fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[], n_segments=50)
        fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[], n_segments=100)

        self.assertEqual(call_count[0], 2)

    def test_cache_cleared_on_confirm_model(self):
        """confirm_model() must wipe _cache_key and _cache_labels."""
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _FakeModel()

        img = np.zeros((10, 20), dtype=np.uint8)
        fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[])
        self.assertIsNotNone(fam._cache_key)
        self.assertIsNotNone(fam._cache_labels)

        # Re-confirm: must call the real confirm_model code path, not just
        # simulate it, so that lines 73-77 are exercised.
        fam.confirm_model()

        self.assertIsNone(fam._cache_key)
        self.assertIsNone(fam._cache_labels)

    # ---- confirm_model ----

    def setUp(self):
        ModelRegistry.model_cache.clear()

    def tearDown(self):
        ModelRegistry.model_cache.clear()

    def test_confirm_model_sets_model_instance(self):
        from core.models.spx import SPX_Tester2D
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.confirm_model()
        self.assertIsInstance(fam.model, SPX_Tester2D)

    def test_confirm_model_invalid_variant_raises(self):
        fam = SPXModelFamily(variant='Unknown')
        with self.assertRaises(ValueError):
            fam.confirm_model()

    # ---- _make_cache_key: unhashable kwargs fallback ----

    def test_make_cache_key_with_unhashable_value_does_not_raise(self):
        """kwargs containing an unhashable value (e.g. a list) must use
        the str() fallback rather than raising TypeError."""
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        img = np.zeros((10, 10), dtype=np.uint8)
        # A list value is unhashable → triggers the except TypeError branch.
        key = fam._make_cache_key(img, {'weights': [1, 2, 3]})
        self.assertIsNotNone(key)

    # ---- on_expand ----

    def test_on_expand_raises_without_model(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        with self.assertRaises(RuntimeError):
            fam.on_expand(img=np.zeros((10, 10)))

    def test_on_expand_raises_without_img(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _FakeModel()
        with self.assertRaises(ValueError):
            fam.on_expand()

    def test_on_expand_returns_label_map(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _FakeModel()
        img = np.zeros((10, 20), dtype=np.uint8)
        result = fam.on_expand(img=img)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (10, 20))

    def test_on_expand_passes_user_params_to_model(self):
        """User params must reach model.forward, not be silently dropped."""
        received = {}

        class _ParamCapture(_FakeModel):
            def forward(self, **kwargs):
                received.update(kwargs)
                return super().forward(**kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _ParamCapture()
        img = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(img=img, n_segments=42, compactness=5)

        self.assertEqual(received.get('n_segments'), 42)
        self.assertEqual(received.get('compactness'), 5)
        # img is passed to model.forward as a kwarg (each model pops it internally)
        self.assertIn('img', received)

    def test_on_expand_uses_label_cache(self):
        """If the label cache is warm for this slice, forward() must not be called again."""
        call_count = [0]
        original_forward = _FakeModel.forward

        class _CountingModel(_FakeModel):
            def forward(self, **kwargs):
                call_count[0] += 1
                return original_forward(self, **kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()
        img = np.zeros((10, 20), dtype=np.uint8)

        # Warm the cache via onRender
        fam.onRender(img=img, pos_points=[[3, 5]], neg_points=[])
        self.assertEqual(call_count[0], 1)

        # Propagate on the same slice — must reuse the cached labels
        fam.on_expand(img=img)
        self.assertEqual(call_count[0], 1, "on_expand should reuse the cache, not call forward() again")

    # ---- interactive stubs ----

    def test_on_enter_interactive_does_not_raise(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.on_enter_interactive()

    def test_on_stop_interactive_does_not_raise(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.on_stop_interactive()


class TestAutoModelFamily(unittest.TestCase):

    def test_on_automatic_segmentation_raises_when_no_model(self):
        fam = AutoModelFamily()
        with self.assertRaises(RuntimeError, msg="Model not confirmed"):
            fam.on_automatic_segmentation()

    def test_on_automatic_segmentation_raises_without_img(self):
        fam = AutoModelFamily()
        fam.model = _FakeModel()
        with self.assertRaises(ValueError, msg="Missing required argument: img"):
            fam.on_automatic_segmentation()

    def test_on_automatic_segmentation_calls_model_forward(self):
        calls = []

        class _TrackingModel:
            def forward(self, **kwargs):
                calls.append(kwargs)
                return None

        fam = AutoModelFamily()
        fam.model = _TrackingModel()
        fam.on_automatic_segmentation(img=np.zeros((4, 4)))
        self.assertEqual(len(calls), 1)
        self.assertIn('img', calls[0])

    def test_on_assign_2d_does_not_raise(self):
        fam = AutoModelFamily()
        fam.on_assign_2d()
        fam.on_assign_2d(extra='ignored')

    def test_on_assign_3d_does_not_raise(self):
        fam = AutoModelFamily()
        fam.on_assign_3d()
        fam.on_assign_3d(extra='ignored')


if __name__ == '__main__':
    unittest.main()
