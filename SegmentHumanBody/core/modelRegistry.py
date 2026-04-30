from .models.spx import SPX_Tester2D, SPX_SLIC2D, SPX_Felzenszwalb2D
from .models.timed_annotator import TimedAnnotatorModel

# Maps registry key → model class.
# Add new models here; do not use globals() lookups.
_MODEL_FACTORIES: dict = {
    'SPX_Tester2D':         SPX_Tester2D,
    'SPX_SLIC2D':           SPX_SLIC2D,
    'SPX_Felzenszwalb2D':   SPX_Felzenszwalb2D,
    'TimedAnnotatorModel':  TimedAnnotatorModel,
}


class ModelRegistry:
    """Lazy-instantiating, session-scoped model cache."""

    model_cache: dict = {}

    @classmethod
    def get_model(cls, key: str):
        """Return a cached model instance, instantiating it on first access."""
        if key not in cls.model_cache:
            cls.model_cache[key] = cls._instantiate(key)
        return cls.model_cache[key]

    @classmethod
    def _instantiate(cls, key: str):
        factory = _MODEL_FACTORIES.get(key)
        if factory is None:
            raise ValueError(
                f"Unknown model key: '{key}'. "
                f"Available: {sorted(_MODEL_FACTORIES)}"
            )
        return factory()

    @classmethod
    def get_param_hint(cls, key: str) -> str:
        """Return PARAM_HINT for a model without instantiating it."""
        factory = _MODEL_FACTORIES.get(key)
        if factory is None:
            return "Model not available. Please select a different model."
        return getattr(factory, 'PARAM_HINT', 'No parameter hint provided.')

    @classmethod
    def register(cls, key: str, factory):
        """Register a new model factory at runtime (e.g. for plugins)."""
        _MODEL_FACTORIES[key] = factory
