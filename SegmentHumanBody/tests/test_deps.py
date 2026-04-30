import os
import unittest

from core._deps import DependencyCheck, _version_ok


class _CacheMixin:
    """Clear DependencyCheck._cache around every test for isolation."""

    def setUp(self):
        DependencyCheck._cache.clear()

    def tearDown(self):
        DependencyCheck._cache.clear()


# ===========================================================================
# check_package
# ===========================================================================

class TestCheckPackage(_CacheMixin, unittest.TestCase):

    def test_available_package_returns_true_empty_message(self):
        ok, msg = DependencyCheck.check_package('numpy')
        self.assertTrue(ok)
        self.assertEqual(msg, '')

    def test_missing_package_returns_false(self):
        ok, _ = DependencyCheck.check_package('_nonexistent_xyz_pkg_')
        self.assertFalse(ok)

    def test_missing_package_message_mentions_import_name(self):
        _, msg = DependencyCheck.check_package('_nonexistent_xyz_pkg_')
        self.assertIn('_nonexistent_xyz_pkg_', msg)

    def test_missing_package_message_contains_install_hint(self):
        _, msg = DependencyCheck.check_package('_nonexistent_xyz_pkg_')
        self.assertIn('pip install', msg)

    def test_display_name_replaces_import_name_in_message(self):
        _, msg = DependencyCheck.check_package(
            '_nonexistent_xyz_pkg_', display_name='my-package'
        )
        self.assertIn('my-package', msg)
        self.assertNotIn('_nonexistent_xyz_pkg_', msg)


# ===========================================================================
# require_package
# ===========================================================================

class TestRequirePackage(_CacheMixin, unittest.TestCase):

    def test_available_package_does_not_raise(self):
        DependencyCheck.require_package('numpy')

    def test_missing_package_raises_import_error(self):
        with self.assertRaises(ImportError):
            DependencyCheck.require_package('_nonexistent_xyz_pkg_')

    def test_import_error_message_contains_package_name(self):
        with self.assertRaises(ImportError) as ctx:
            DependencyCheck.require_package('_nonexistent_xyz_pkg_')
        self.assertIn('_nonexistent_xyz_pkg_', str(ctx.exception))

    def test_display_name_appears_in_import_error(self):
        with self.assertRaises(ImportError) as ctx:
            DependencyCheck.require_package(
                '_nonexistent_xyz_pkg_', display_name='my-nice-name'
            )
        self.assertIn('my-nice-name', str(ctx.exception))


# ===========================================================================
# check_file
# ===========================================================================

class TestCheckFile(_CacheMixin, unittest.TestCase):

    def test_existing_file_returns_true_empty_message(self):
        path = os.path.abspath(__file__)
        ok, msg = DependencyCheck.check_file(path)
        self.assertTrue(ok)
        self.assertEqual(msg, '')

    def test_missing_file_returns_false(self):
        ok, _ = DependencyCheck.check_file('/no/such/file_xyz.pth')
        self.assertFalse(ok)

    def test_missing_file_message_contains_path(self):
        path = '/no/such/file_xyz.pth'
        _, msg = DependencyCheck.check_file(path)
        self.assertIn(path, msg)

    def test_display_name_appears_in_missing_file_message(self):
        _, msg = DependencyCheck.check_file(
            '/no/such/file_xyz.pth', display_name='model-weights'
        )
        self.assertIn('model-weights', msg)

    def test_directory_is_not_a_file(self):
        ok, _ = DependencyCheck.check_file(os.path.dirname(os.path.abspath(__file__)))
        self.assertFalse(ok)


# ===========================================================================
# require_file
# ===========================================================================

class TestRequireFile(_CacheMixin, unittest.TestCase):

    def test_existing_file_does_not_raise(self):
        path = os.path.abspath(__file__)
        DependencyCheck.require_file(path)

    def test_missing_file_raises_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            DependencyCheck.require_file('/no/such/file_xyz.pth')

    def test_file_not_found_message_contains_path(self):
        path = '/no/such/file_xyz.pth'
        with self.assertRaises(FileNotFoundError) as ctx:
            DependencyCheck.require_file(path)
        self.assertIn(path, str(ctx.exception))

    def test_display_name_in_file_not_found_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            DependencyCheck.require_file('/no/such/file_xyz.pth', display_name='weights')
        self.assertIn('weights', str(ctx.exception))


# ===========================================================================
# Caching behaviour
# ===========================================================================

class TestCaching(_CacheMixin, unittest.TestCase):

    def test_package_result_stored_in_cache_after_first_call(self):
        DependencyCheck.check_package('numpy')
        self.assertIn(('pkg', 'numpy', None), DependencyCheck._cache)

    def test_file_result_stored_in_cache_after_first_call(self):
        path = os.path.abspath(__file__)
        DependencyCheck.check_file(path)
        self.assertIn(('file', path), DependencyCheck._cache)

    def test_poisoned_cache_entry_is_returned_without_re_probing(self):
        key = ('pkg', 'numpy', None)
        DependencyCheck._cache[key] = 'injected error'
        ok, msg = DependencyCheck.check_package('numpy')
        self.assertFalse(ok)
        self.assertEqual(msg, 'injected error')

    def test_require_package_reads_poisoned_cache_and_raises(self):
        key = ('pkg', 'numpy', None)
        DependencyCheck._cache[key] = 'injected error'
        with self.assertRaises(ImportError):
            DependencyCheck.require_package('numpy')

    def test_different_min_versions_use_separate_cache_keys(self):
        DependencyCheck.check_package('numpy', min_version='1.0')
        DependencyCheck.check_package('numpy', min_version='2.0')
        self.assertIn(('pkg', 'numpy', '1.0'), DependencyCheck._cache)
        self.assertIn(('pkg', 'numpy', '2.0'), DependencyCheck._cache)

    def test_no_version_and_versioned_check_use_separate_keys(self):
        DependencyCheck.check_package('numpy')
        DependencyCheck.check_package('numpy', min_version='1.0')
        self.assertIn(('pkg', 'numpy', None), DependencyCheck._cache)
        self.assertIn(('pkg', 'numpy', '1.0'), DependencyCheck._cache)

    def test_missing_package_result_is_also_cached(self):
        DependencyCheck.check_package('_nonexistent_xyz_pkg_')
        key = ('pkg', '_nonexistent_xyz_pkg_', None)
        self.assertIn(key, DependencyCheck._cache)
        self.assertIsNotNone(DependencyCheck._cache[key])


# ===========================================================================
# _version_ok helper
# ===========================================================================

class TestVersionOk(unittest.TestCase):

    def test_equal_versions_are_ok(self):
        self.assertTrue(_version_ok('1.0.0', '1.0.0'))

    def test_higher_major_is_ok(self):
        self.assertTrue(_version_ok('2.0.0', '1.9.9'))

    def test_higher_minor_is_ok(self):
        self.assertTrue(_version_ok('0.21.3', '0.19.0'))

    def test_higher_patch_is_ok(self):
        self.assertTrue(_version_ok('1.0.5', '1.0.4'))

    def test_lower_major_is_not_ok(self):
        self.assertFalse(_version_ok('0.9.0', '1.0.0'))

    def test_lower_minor_is_not_ok(self):
        self.assertFalse(_version_ok('0.18.0', '0.19.0'))

    def test_lower_patch_is_not_ok(self):
        self.assertFalse(_version_ok('1.0.3', '1.0.4'))

    def test_two_part_version_comparison(self):
        self.assertTrue(_version_ok('2.1', '2.0'))
        self.assertFalse(_version_ok('1.9', '2.0'))

    def test_version_ok_with_satisfied_numpy_min(self):
        # numpy is always available; 0.1 is always satisfied
        ok, _ = DependencyCheck.check_package('numpy', min_version='0.1')
        DependencyCheck._cache.clear()
        self.assertTrue(ok)


# ===========================================================================
# Integration: SPX model __init__ uses DependencyCheck
# ===========================================================================

class TestSPXModelsUseDependencyCheck(_CacheMixin, unittest.TestCase):
    """Verify that SPX model constructors surface dep failures via DependencyCheck."""

    def test_slic_init_populates_cache_for_skimage(self):
        try:
            from core.models.spx import SPX_SLIC2D
            SPX_SLIC2D()
        except ImportError:
            pass  # skimage absent — ImportError expected; cache should still be set
        self.assertIn(('pkg', 'skimage', None), DependencyCheck._cache)

    def test_felzenszwalb_init_populates_cache_for_skimage(self):
        try:
            from core.models.spx import SPX_Felzenszwalb2D
            SPX_Felzenszwalb2D()
        except ImportError:
            pass
        self.assertIn(('pkg', 'skimage', None), DependencyCheck._cache)

    def test_slic_second_construction_hits_cache(self):
        """Both SLIC models share the same skimage cache key."""
        try:
            from core.models.spx import SPX_SLIC2D
            SPX_SLIC2D()
        except ImportError:
            pass
        # Poison and verify the second construction reads the poisoned value.
        DependencyCheck._cache[('pkg', 'skimage', None)] = 'injected'
        from core.models.spx import SPX_SLIC2D
        with self.assertRaises(ImportError) as ctx:
            SPX_SLIC2D()
        self.assertIn('injected', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
