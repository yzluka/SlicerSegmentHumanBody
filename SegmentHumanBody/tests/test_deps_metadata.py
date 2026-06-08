"""Tests for DependencyCheck.check_distribution (metadata-only dep probe)."""
import unittest
from unittest.mock import patch, MagicMock
from core._deps import DependencyCheck


class TestCheckDistribution(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test so probes are fresh.
        DependencyCheck._cache.clear()

    def tearDown(self):
        DependencyCheck._cache.clear()

    def test_installed_package_returns_true(self):
        # numpy is always present in Slicer's Python environment.
        ok, msg = DependencyCheck.check_distribution('numpy')
        self.assertTrue(ok)
        self.assertEqual(msg, '')

    def test_missing_package_returns_false(self):
        ok, msg = DependencyCheck.check_distribution('this-package-does-not-exist-xyzzy')
        self.assertFalse(ok)
        self.assertIn('this-package-does-not-exist-xyzzy', msg)

    def test_result_is_cached(self):
        # Prime the cache.
        DependencyCheck.check_distribution('numpy')
        # Monkeypatch distribution to raise — cached result must be returned.
        from importlib.metadata import PackageNotFoundError
        with patch('importlib.metadata.distribution', side_effect=PackageNotFoundError('numpy')):
            ok, _ = DependencyCheck.check_distribution('numpy')
        # Should still be True because the result came from cache, not the new probe.
        self.assertTrue(ok)

    def test_min_version_satisfied(self):
        ok, msg = DependencyCheck.check_distribution('numpy', min_version='1.0')
        self.assertTrue(ok)
        self.assertEqual(msg, '')

    def test_min_version_too_high_returns_false(self):
        ok, msg = DependencyCheck.check_distribution('numpy', min_version='9999.0')
        self.assertFalse(ok)
        self.assertIn('9999.0', msg)

    def test_does_not_import_unrelated_package(self):
        """Probing scikit-image must not import torch."""
        import sys
        had_torch = 'torch' in sys.modules
        DependencyCheck.check_distribution('scikit-image')
        if not had_torch:
            self.assertNotIn('torch', sys.modules)

    def test_cache_key_includes_min_version(self):
        """check_distribution('numpy') and check_distribution('numpy', min_version='1.0')
        must be cached separately."""
        DependencyCheck.check_distribution('numpy')
        DependencyCheck.check_distribution('numpy', min_version='1.0')
        # Two distinct cache entries.
        count = sum(1 for k in DependencyCheck._cache if k[1] == 'numpy')
        self.assertEqual(count, 2)
