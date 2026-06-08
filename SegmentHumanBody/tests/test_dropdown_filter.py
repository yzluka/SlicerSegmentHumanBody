"""Tests for dependency-filtered family/variant dropdowns."""
import unittest
from unittest.mock import patch
from core._deps import DependencyCheck
from core.modelFamilies import SPXModelFamily, SAMFamily, TimedAnnotatorFamily, FAMILY_REGISTRY
from core.modelRegistry import ModelRegistry


def _check_dist_ok(dist_name, *, min_version=None):
    return True, ''


def _check_dist_fail(dist_name, *, min_version=None):
    return False, f"Missing distribution: '{dist_name}'"


class TestModelAvailability(unittest.TestCase):

    def setUp(self):
        DependencyCheck._cache.clear()
        ModelRegistry.model_cache.clear()

    def tearDown(self):
        DependencyCheck._cache.clear()
        ModelRegistry.model_cache.clear()

    def test_tester2d_always_available(self):
        """SPX_Tester2D has no distribution requirements — always available."""
        self.assertTrue(ModelRegistry.is_model_available('SPX_Tester2D'))

    def test_slic2d_available_when_scikit_image_present(self):
        with patch.object(DependencyCheck, 'check_distribution', side_effect=_check_dist_ok):
            DependencyCheck._cache.clear()
            self.assertTrue(ModelRegistry.is_model_available('SPX_SLIC2D'))

    def test_slic2d_unavailable_when_scikit_image_missing(self):
        DependencyCheck._cache[('dist', 'scikit-image', None)] = "Missing distribution: 'scikit-image'"
        self.assertFalse(ModelRegistry.is_model_available('SPX_SLIC2D'))

    def test_felzenszwalb_unavailable_when_scikit_image_missing(self):
        DependencyCheck._cache[('dist', 'scikit-image', None)] = "Missing distribution: 'scikit-image'"
        self.assertFalse(ModelRegistry.is_model_available('SPX_Felzenszwalb2D'))

    def test_unknown_key_returns_false(self):
        self.assertFalse(ModelRegistry.is_model_available('NonExistentModel'))


class TestSPXVariantFiltering(unittest.TestCase):
    """Simulate dropdown population: scikit-image missing → only Naive_Grid-2D remains."""

    def setUp(self):
        DependencyCheck._cache.clear()
        ModelRegistry.model_cache.clear()

    def tearDown(self):
        DependencyCheck._cache.clear()
        ModelRegistry.model_cache.clear()

    def _available_variants(self, family_cls):
        variants = []
        for variant, model_key in (family_cls.MODEL_MAP or {}).items():
            if ModelRegistry.is_model_available(model_key):
                variants.append(variant)
        return variants

    def test_all_three_variants_when_skimage_present(self):
        with patch.object(DependencyCheck, 'check_distribution', return_value=(True, '')):
            DependencyCheck._cache.clear()
            variants = self._available_variants(SPXModelFamily)
        self.assertIn('Naive_Grid-2D', variants)
        self.assertIn('SLIC-2D', variants)
        self.assertIn('Felzenszwalb-2D', variants)
        self.assertEqual(len(variants), 3)

    def test_only_naive_grid_when_skimage_missing(self):
        DependencyCheck._cache[('dist', 'scikit-image', None)] = "Missing"
        variants = self._available_variants(SPXModelFamily)
        self.assertEqual(variants, ['Naive_Grid-2D'])

    def test_timed_annotator_family_has_no_model_map(self):
        """TimedAnnotatorFamily has empty VARIANTS — no model-map filter applies."""
        self.assertEqual(TimedAnnotatorFamily.VARIANTS, [])

    def test_requires_distributions_on_sam_family(self):
        reqs = SAMFamily.REQUIRES_DISTRIBUTIONS
        self.assertTrue(len(reqs) >= 1)
        dist_names = [r[0] for r in reqs]
        self.assertIn('torch', dist_names)

    def test_spx_tester_has_empty_requires(self):
        from core.models.spx import SPX_Tester2D
        self.assertEqual(SPX_Tester2D.REQUIRES_DISTRIBUTIONS, ())

    def test_spx_slic_requires_scikit_image(self):
        from core.models.spx import SPX_SLIC2D
        dist_names = [r[0] for r in SPX_SLIC2D.REQUIRES_DISTRIBUTIONS]
        self.assertIn('scikit-image', dist_names)
