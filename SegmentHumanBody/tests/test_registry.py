import unittest

from core.modelRegistry import ModelRegistry, _MODEL_FACTORIES
from core.models.spx import SPX_Tester2D, SPX_SLIC2D, SPX_Felzenszwalb2D


class TestModelRegistry(unittest.TestCase):

    def setUp(self):
        # Isolate each test from session-level cache state.
        ModelRegistry.model_cache.clear()

    def tearDown(self):
        ModelRegistry.model_cache.clear()
        # Remove any keys registered during tests
        for key in list(_MODEL_FACTORIES.keys()):
            if key.startswith('_test_'):
                del _MODEL_FACTORIES[key]

    # --- get_model ---

    def test_get_model_returns_spx_tester_instance(self):
        model = ModelRegistry.get_model('SPX_Tester2D')
        self.assertIsInstance(model, SPX_Tester2D)

    def test_get_model_returns_spx_slic_instance(self):
        try:
            model = ModelRegistry.get_model('SPX_SLIC2D')
            self.assertIsInstance(model, SPX_SLIC2D)
        except ImportError:
            self.skipTest("scikit-image not installed")

    def test_get_model_returns_felzenszwalb_instance(self):
        try:
            model = ModelRegistry.get_model('SPX_Felzenszwalb2D')
            self.assertIsInstance(model, SPX_Felzenszwalb2D)
        except ImportError:
            self.skipTest("scikit-image not installed")

    def test_get_model_caches_instance(self):
        first = ModelRegistry.get_model('SPX_Tester2D')
        second = ModelRegistry.get_model('SPX_Tester2D')
        self.assertIs(first, second)

    def test_get_model_unknown_key_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            ModelRegistry.get_model('DoesNotExist')
        self.assertIn('DoesNotExist', str(ctx.exception))

    # --- _instantiate ---

    def test_instantiate_model_unknown_key_raises_value_error(self):
        with self.assertRaises(ValueError):
            ModelRegistry.instantiate_model('Bogus')

    def test_instantiate_model_returns_new_instance_each_call(self):
        a = ModelRegistry.instantiate_model('SPX_Tester2D')
        b = ModelRegistry.instantiate_model('SPX_Tester2D')
        self.assertIsNot(a, b)

    # --- get_param_hint ---

    def test_get_param_hint_returns_correct_hint(self):
        hint = ModelRegistry.get_param_hint('SPX_Tester2D')
        self.assertEqual(hint, SPX_Tester2D.PARAM_HINT)

    def test_get_param_hint_unknown_key_returns_fallback_string(self):
        hint = ModelRegistry.get_param_hint('NoSuchModel')
        self.assertIsInstance(hint, str)
        self.assertGreater(len(hint), 0)

    def test_get_param_hint_does_not_instantiate_model(self):
        """Hint must be readable without putting anything into the cache."""
        ModelRegistry.get_param_hint('SPX_Tester2D')
        self.assertNotIn('SPX_Tester2D', ModelRegistry.model_cache)

    # --- register ---

    def test_register_adds_new_factory(self):
        class _DummyModel:
            PARAM_HINT = 'x=1'
            def forward(self, **kwargs): return None

        ModelRegistry.register('_test_dummy', _DummyModel)
        model = ModelRegistry.get_model('_test_dummy')
        self.assertIsInstance(model, _DummyModel)

    def test_register_hint_visible_before_instantiation(self):
        class _HintModel:
            PARAM_HINT = 'a=42'

        ModelRegistry.register('_test_hint', _HintModel)
        self.assertEqual(ModelRegistry.get_param_hint('_test_hint'), 'a=42')


if __name__ == '__main__':
    unittest.main()
