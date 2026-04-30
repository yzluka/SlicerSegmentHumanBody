"""Process-scoped dependency cache for lazy, single-probe dependency checking.

Usage in a model ``__init__``::

    from .._deps import DependencyCheck

    class MyModel(SPXModel):
        def __init__(self):
            DependencyCheck.require_package('skimage', display_name='scikit-image')
            from skimage.segmentation import slic
            self._slic = slic

The first ``require_package`` call probes the environment; every subsequent
call in the same process is O(1) — the result is read from ``_cache``.
"""

import importlib


def _version_ok(installed: str, minimum: str) -> bool:
    """Return True if *installed* >= *minimum* (dotted-integer comparison)."""
    def _t(v: str):
        try:
            return tuple(int(x) for x in v.split('.'))
        except ValueError:
            return (0,)
    return _t(installed) >= _t(minimum)


class DependencyCheck:
    """Lazy, process-scoped checker for Python packages and file paths.

    All results are stored in the class-level ``_cache`` dict so that repeated
    calls for the same dependency within a process are O(1) after the first probe.

    Callers use the ``require_*`` methods (raise on failure) or ``check_*``
    methods (return ``(ok: bool, message: str)`` without raising).
    """

    # (kind, *key_parts) → None (ok) | str (error message)
    _cache: dict = {}

    # ------------------------------------------------------------------
    # Public API — raising variants
    # ------------------------------------------------------------------

    @classmethod
    def require_package(
        cls,
        import_name: str,
        *,
        display_name: str | None = None,
        min_version: str | None = None,
    ) -> None:
        """Raise ``ImportError`` if *import_name* is missing or below *min_version*.

        Parameters
        ----------
        import_name:
            Name used with ``importlib.import_module`` (e.g. ``'skimage'``).
        display_name:
            Human-readable install name for error messages
            (e.g. ``'scikit-image'``).  Defaults to *import_name*.
        min_version:
            Minimum acceptable version string, e.g. ``'0.19'`` or ``'2.0.1'``.
            Checked via ``importlib.metadata``; ignored if the metadata is
            unavailable.
        """
        msg = cls._package_result(import_name, display_name or import_name, min_version)
        if msg:
            raise ImportError(msg)

    @classmethod
    def require_file(
        cls,
        path: str,
        *,
        display_name: str | None = None,
    ) -> None:
        """Raise ``FileNotFoundError`` if *path* does not exist on disk.

        Parameters
        ----------
        path:
            Absolute or relative filesystem path to check.
        display_name:
            Short label shown in error messages (e.g. ``'model weights'``).
            Defaults to *path*.
        """
        msg = cls._file_result(path, display_name or path)
        if msg:
            raise FileNotFoundError(msg)

    # ------------------------------------------------------------------
    # Public API — non-raising variants
    # ------------------------------------------------------------------

    @classmethod
    def check_package(
        cls,
        import_name: str,
        *,
        display_name: str | None = None,
        min_version: str | None = None,
    ) -> tuple[bool, str]:
        """Return ``(ok, message)`` without raising.

        *message* is an empty string when *ok* is ``True``.
        """
        msg = cls._package_result(import_name, display_name or import_name, min_version)
        return msg is None, msg or ''

    @classmethod
    def check_file(
        cls,
        path: str,
        *,
        display_name: str | None = None,
    ) -> tuple[bool, str]:
        """Return ``(ok, message)`` without raising."""
        msg = cls._file_result(path, display_name or path)
        return msg is None, msg or ''

    # ------------------------------------------------------------------
    # Internal — cache coordination
    # ------------------------------------------------------------------

    @classmethod
    def _package_result(cls, import_name: str, display_name: str, min_version) -> str | None:
        key = ('pkg', import_name, min_version)
        if key not in cls._cache:
            cls._cache[key] = cls._probe_package(import_name, display_name, min_version)
        return cls._cache[key]

    @classmethod
    def _file_result(cls, path: str, display_name: str) -> str | None:
        key = ('file', path)
        if key not in cls._cache:
            cls._cache[key] = cls._probe_file(path, display_name)
        return cls._cache[key]

    # ------------------------------------------------------------------
    # Internal — actual probes (called at most once per key per process)
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_package(import_name: str, display_name: str, min_version) -> str | None:
        try:
            importlib.import_module(import_name)
        except ImportError:
            return f"Missing: '{display_name}'. Install with: pip install {display_name}"

        if min_version is not None:
            try:
                from importlib.metadata import version
                installed = version(display_name)
                if not _version_ok(installed, min_version):
                    return (
                        f"'{display_name}' {installed} is installed but "
                        f">={min_version} is required. "
                        f"Upgrade with: pip install --upgrade {display_name}"
                    )
            except Exception:
                pass  # can't determine version — treat as satisfied

        return None

    @staticmethod
    def _probe_file(path: str, display_name: str) -> str | None:
        import os
        if not os.path.isfile(path):
            return f"Required file not found: {display_name!r} (path: {path})"
        return None
