from .models.default import IdentityModel
from .models.spx import SPX_Tester2D, SPX_SLIC2D, SPX_Felzenszwalb2D
from .models.timed_annotator import TimedAnnotatorModel
from ._deps import DependencyCheck

# Maps registry key → model class.
# Add new models here; do not use globals() lookups.
_MODEL_FACTORIES: dict = {
    'Identity':             IdentityModel,
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
    def is_model_available(cls, key: str) -> bool:
        """Return True if all distributions declared by the model are present.

        Uses metadata-only probes (no import). Models with no
        ``REQUIRES_DISTRIBUTIONS`` attribute are always available.
        """
        factory = _MODEL_FACTORIES.get(key)
        if factory is None:
            return False
        for dist_name, min_ver in getattr(factory, 'REQUIRES_DISTRIBUTIONS', ()):
            ok, _ = DependencyCheck.check_distribution(dist_name, min_version=min_ver)
            if not ok:
                return False
        return True

    @classmethod
    def register(cls, key: str, factory):
        """Register a new model factory at runtime (e.g. for plugins)."""
        _MODEL_FACTORIES[key] = factory
