import unittest
import numpy as np

from core.modelFamilies import (
    BaseModelFamily, DefaultFamily, SAMFamily, SPXModelFamily, AutoModelFamily,
    TimedAnnotatorFamily, FAMILY_REGISTRY,
)
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


class _FakeVolumeNode:
    """Minimal stub for a Slicer vtkMRMLScalarVolumeNode used in cache-key tests."""

    def __init__(self, node_id='vol-1', mtime=100):
        self._id = node_id
        self._mtime = mtime

    def GetID(self):
        return self._id

    def GetMTime(self):
        return self._mtime

    def bump_mtime(self):
        self._mtime += 1


class TestBaseModelFamily(unittest.TestCase):

    def setUp(self):
        ModelRegistry.model_cache.clear()

    def tearDown(self):
        ModelRegistry.model_cache.clear()

    def test_confirm_model_no_variant_raises(self):
        fam = BaseModelFamily(variant=None)
        with self.assertRaises(ValueError):
            fam.confirm_model()

    def test_confirm_model_sets_model_from_registry(self):
        fam = BaseModelFamily(variant='SPX_Tester2D')
        fam.confirm_model()
        self.assertIsNotNone(fam.model)


class TestDefaultFamily(unittest.TestCase):

    def setUp(self):
        ModelRegistry.model_cache.clear()

    def tearDown(self):
        ModelRegistry.model_cache.clear()

    def test_registry_exposes_basic_family(self):
        self.assertIs(FAMILY_REGISTRY['Basic'], DefaultFamily)

    def test_registry_does_not_expose_default_or_none(self):
        self.assertNotIn('Default', FAMILY_REGISTRY)
        self.assertNotIn('None', FAMILY_REGISTRY)

    def test_default_family_declares_basic_variant(self):
        self.assertEqual(DefaultFamily.VARIANTS, ['Basic'])

    def test_confirm_model_loads_identity_model(self):
        fam = DefaultFamily(variant='Basic')
        fam.confirm_model()
        self.assertIsNotNone(fam.model)

    def test_identity_on_render_returns_input_image(self):
        fam = DefaultFamily(variant='Basic')
        fam.confirm_model()
        img = np.arange(9, dtype=np.uint8).reshape(3, 3)
        self.assertIs(fam.onRender(img=img), img)

    def test_identity_on_expand_returns_input_image(self):
        fam = DefaultFamily(variant='Basic')
        fam.confirm_model()
        img = np.arange(4, dtype=np.uint8).reshape(2, 2)
        self.assertIs(fam.on_expand(img=img), img)


class TestSAMFamily(unittest.TestCase):
    """SAMFamily methods are intentional stubs (not yet implemented).
    Tests confirm they exist, accept **kwargs, and do not raise."""

    def setUp(self):
        self.fam = SAMFamily(variant='SAM-VIT-H')

    def test_get_requested_mask_does_not_raise(self):
        self.fam.get_requested_mask()
        self.fam.get_requested_mask(mask_index=0)

    def test_onrender_stub_does_not_raise_and_returns_none(self):
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

    # ---- on_expand cache ----

    def _make_family_with_fake_model(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _FakeModel()
        return fam

    def test_cache_hit_skips_forward(self):
        """Same volume/axis/slice/kwargs → model.forward called exactly once."""
        call_count = [0]
        original_forward = _FakeModel.forward

        class _CountingModel(_FakeModel):
            def forward(self, **kwargs):
                call_count[0] += 1
                return original_forward(self, **kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)

        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)

        self.assertEqual(call_count[0], 1, "forward() should be called once for repeated calls on the same slice")

    def test_cache_stable_on_mtime_bump(self):
        """MTime bump alone must NOT bust the cache.
        Slicer's display pipeline increments MTime after every
        modifySelectedSegmentByLabelmap call, so including it would force
        SLIC/Felzenszwalb to re-run on every stroke (~1-2 s each)."""
        call_count = [0]
        original_forward = _FakeModel.forward

        class _CountingModel(_FakeModel):
            def forward(self, **kwargs):
                call_count[0] += 1
                return original_forward(self, **kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)

        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        vol.bump_mtime()  # rendering pipeline side-effect — must not invalidate
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)

        self.assertEqual(call_count[0], 1)

    def test_cache_miss_on_different_kwargs(self):
        """Same volume/axis/slice but different user params → forward called again."""
        call_count = [0]
        original_forward = _FakeModel.forward

        class _CountingModel(_FakeModel):
            def forward(self, **kwargs):
                call_count[0] += 1
                return original_forward(self, **kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)

        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img, n_segments=50)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img, n_segments=100)

        self.assertEqual(call_count[0], 2)

    def test_cache_cleared_on_confirm_model(self):
        """confirm_model() must wipe _cache_key and _cache_labels."""
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _FakeModel()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        self.assertIsNotNone(fam._cache_key)
        self.assertIsNotNone(fam._cache_labels)

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
        vol = _FakeVolumeNode()
        # A list value is unhashable → triggers the except TypeError branch.
        key = fam._make_cache_key(vol, 'ax', 0, {'weights': [1, 2, 3]})
        self.assertIsNotNone(key)

    # ---- on_expand ----

    def test_on_expand_raises_without_model(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        vol = _FakeVolumeNode()
        with self.assertRaises(RuntimeError):
            fam.on_expand(volume_node=vol, axis='ax', slice_idx=0, img=np.zeros((10, 10)))

    def test_on_expand_raises_without_img(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _FakeModel()
        vol = _FakeVolumeNode()
        with self.assertRaises(ValueError):
            fam.on_expand(volume_node=vol, axis='ax', slice_idx=0)

    def test_on_expand_returns_label_map(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _FakeModel()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)
        result = fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
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
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5,
                      img=img, n_segments=42, compactness=5)

        self.assertEqual(received.get('n_segments'), 42)
        self.assertEqual(received.get('compactness'), 5)
        # img is passed to model.forward as a kwarg (each model pops it internally)
        self.assertIn('img', received)

    def test_on_expand_uses_label_cache(self):
        """Repeated on_expand calls on the same slice must call forward() only once."""
        call_count = [0]
        original_forward = _FakeModel.forward

        class _CountingModel(_FakeModel):
            def forward(self, **kwargs):
                call_count[0] += 1
                return original_forward(self, **kwargs)

        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)

        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        self.assertEqual(call_count[0], 1)

        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        self.assertEqual(call_count[0], 1, "on_expand should reuse the cache, not call forward() again")



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


class TestTimedAnnotatorFamily(unittest.TestCase):
    """Pure-Python tests for TimedAnnotatorFamily (no slicer/vtk)."""

    def setUp(self):
        from core.modelRegistry import ModelRegistry
        ModelRegistry.model_cache.clear()
        self.fam = TimedAnnotatorFamily()
        self.fam.confirm_model()   # loads TimedAnnotatorModel into self.fam.model

    def tearDown(self):
        from core.modelRegistry import ModelRegistry
        ModelRegistry.model_cache.clear()

    # --- basic properties ---

    def test_variants_is_empty(self):
        self.assertEqual(TimedAnnotatorFamily.VARIANTS, [])

    def test_confirm_model_loads_timed_annotator_model(self):
        from core.models.timed_annotator import TimedAnnotatorModel
        self.assertIsInstance(self.fam.model, TimedAnnotatorModel)

    def test_visible_buttons_contains_export(self):
        self.assertIn('exportAnnotationLogButton', TimedAnnotatorFamily.VISIBLE_BUTTONS)

    # --- on_segment_created ---

    def test_on_segment_created_records_entry(self):
        self.fam.on_segment_created('seg-1', 'Segment_1')
        self.assertEqual(len(self.fam.model._log), 1)
        entry = self.fam.model._log[0]
        self.assertEqual(entry['type'], 'segment')
        self.assertEqual(entry['segment_id'], 'seg-1')
        self.assertEqual(entry['seg_name'], 'Segment_1')
        self.assertIn('timestamp', entry)

    def test_on_segment_created_timestamp_is_iso(self):
        self.fam.on_segment_created('seg-1', 'Segment_1')
        ts = self.fam.model._log[0]['timestamp']
        from datetime import datetime
        # Should not raise — ISO-8601 parseable
        datetime.fromisoformat(ts)

    # --- on_point_confirmed (pure-Python portion) ---
    # _mirror_to_node imports slicer so we skip it here; we test the log state
    # by injecting a stub that bypasses the slicer call.

    def _add_point_bypassing_mirror(self, fam, seg_id, ras, cp_id, mirror_cp_id='m-1'):
        """Directly append a point log entry without calling _mirror_to_node."""
        from datetime import datetime
        ts = datetime.now().isoformat()
        fam.model._log.append({
            'type':         'point',
            'segment_id':   seg_id,
            'coord_ras':    list(ras),
            'timestamp':    ts,
            'cp_id':        cp_id,
            'mirror_cp_id': mirror_cp_id,
        })
        fam.model._point_history[mirror_cp_id] = {
            'segment_id': seg_id,
            'versions':   [{'coord_ras': list(ras), 'timestamp': ts}],
            'alive':      True,
        }

    def test_log_grows_on_point_append(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1, 2, 3], 'cp-1')
        self.assertEqual(len(self.fam.model._log), 1)

    def test_export_data_returns_nested_dict(self):
        data = self.fam.export_data()
        self.assertIsInstance(data, dict)
        self.assertIn('segments', data)

    def test_export_data_returns_only_point_entries(self):
        self.fam.on_segment_created('seg-1', 'Segment_1')
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 2.0, 3.0], 'cp-1')
        data = self.fam.export_data()
        self.assertEqual(len(data['segments']['seg-1']['points']), 1)

    def test_export_data_segment_has_seg_name(self):
        self.fam.on_segment_created('seg-1', 'Segment_1')
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 2.0, 3.0], 'cp-1')
        seg = self.fam.export_data()['segments']['seg-1']
        self.assertEqual(seg['seg_name'], 'Segment_1')

    def test_export_data_falls_back_to_segment_id_when_no_name(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 2.0, 3.0], 'cp-1')
        seg = self.fam.export_data()['segments']['seg-1']
        self.assertEqual(seg['seg_name'], 'seg-1')

    def test_export_data_points_keyed_by_mirror_cp_id(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 2.0, 3.0], 'cp-1', 'm-1')
        points = self.fam.export_data()['segments']['seg-1']['points']
        self.assertIsInstance(points, dict)
        self.assertIn('m-1', points)

    def test_export_data_excludes_internal_fields(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 2.0, 3.0], 'cp-1', 'm-1')
        point = next(iter(self.fam.export_data()['segments']['seg-1']['points'].values()))
        self.assertNotIn('cp_id', point)
        self.assertNotIn('mirror_cp_id', point)
        self.assertNotIn('type', point)
        self.assertNotIn('segment_id', point)

    def test_export_data_contains_required_point_fields(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 2.0, 3.0], 'cp-1', 'm-1')
        point = next(iter(self.fam.export_data()['segments']['seg-1']['points'].values()))
        self.assertIn('alive', point)
        self.assertIn('coord_ras', point)
        self.assertIn('coord_ijk', point)
        self.assertIn('timestamp', point)

    def test_export_data_alive_is_true_for_existing_point(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 2.0, 3.0], 'cp-1', 'm-1')
        point = next(iter(self.fam.export_data()['segments']['seg-1']['points'].values()))
        self.assertTrue(point['alive'])

    def test_export_data_coord_ras_is_versioned_list(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 2.0, 3.0], 'cp-1', 'm-1')
        point = next(iter(self.fam.export_data()['segments']['seg-1']['points'].values()))
        self.assertIsInstance(point['coord_ras'], list)
        self.assertIsInstance(point['coord_ras'][0], list)

    def test_export_data_coord_ijk_is_none_when_no_volume(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 2.0, 3.0], 'cp-1', 'm-1')
        point = next(iter(self.fam.export_data()['segments']['seg-1']['points'].values()))
        self.assertIsNone(point['coord_ijk'][0])

    def test_export_data_coord_ras_matches(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.5, 2.5, 3.5], 'cp-1', 'm-1')
        point = next(iter(self.fam.export_data()['segments']['seg-1']['points'].values()))
        self.assertEqual(point['coord_ras'][0], [1.5, 2.5, 3.5])

    def test_export_data_two_segments_both_present(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 0.0, 0.0], 'cp-1')
        self._add_point_bypassing_mirror(self.fam, 'seg-2', [2.0, 0.0, 0.0], 'cp-2')
        segs = self.fam.export_data()['segments']
        self.assertIn('seg-1', segs)
        self.assertIn('seg-2', segs)

    def test_export_data_deleted_point_still_exported_with_alive_false(self):
        from datetime import datetime
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1.0, 0.0, 0.0], 'cp-1', 'm-1')
        # Simulate deletion: mark history as deleted, remove from log
        self.fam.model._point_history['m-1']['versions'].append(
            {'coord_ras': None, 'timestamp': datetime.now().isoformat()}
        )
        self.fam.model._point_history['m-1']['alive'] = False
        self.fam.model._log = [e for e in self.fam.model._log if e.get('mirror_cp_id') != 'm-1']
        points = self.fam.export_data()['segments']['seg-1']['points']
        self.assertIn('m-1', points)
        self.assertFalse(points['m-1']['alive'])
        self.assertIsNone(points['m-1']['coord_ras'][-1])

    def test_export_data_deleted_point_has_none_as_last_coord_ras(self):
        from datetime import datetime
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [5.0, 0.0, 0.0], 'cp-1', 'm-1')
        self.fam.model._point_history['m-1']['versions'].append(
            {'coord_ras': None, 'timestamp': datetime.now().isoformat()}
        )
        self.fam.model._point_history['m-1']['alive'] = False
        self.fam.model._log = [e for e in self.fam.model._log if e.get('mirror_cp_id') != 'm-1']
        point = self.fam.export_data()['segments']['seg-1']['points']['m-1']
        self.assertEqual(len(point['coord_ras']), 2)       # v0 original, v1 deletion
        self.assertEqual(point['coord_ras'][0], [5.0, 0.0, 0.0])
        self.assertIsNone(point['coord_ras'][-1])

    # --- on_point_undone ---

    def _make_stub_mirror_node(self):
        """Return a stub that tracks RemoveNthControlPoint calls."""
        class _StubNode:
            def __init__(self, cp_id, idx):
                self._cp_id = cp_id
                self._idx = idx
                self.removed = []

            def GetControlPointIndexByID(self, cp_id):
                return self._idx if cp_id == self._cp_id else -1

            def RemoveNthControlPoint(self, idx):
                self.removed.append(idx)

        return _StubNode

    def test_on_point_undone_removes_log_entry(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1, 2, 3], 'cp-1', 'm-1')
        self.fam.on_point_undone('cp-1')
        point_entries = [e for e in self.fam.model._log if e.get('type') == 'point']
        self.assertEqual(len(point_entries), 0)

    def test_on_point_undone_removes_from_mirror_node(self):
        _Node = self._make_stub_mirror_node()
        stub = _Node('m-1', 0)
        self.fam.model._nodes['seg-1'] = stub
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1, 2, 3], 'cp-1', 'm-1')
        self.fam.on_point_undone('cp-1')
        self.assertEqual(stub.removed, [0])

    def test_on_point_undone_unknown_cp_id_is_no_op(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1, 2, 3], 'cp-1', 'm-1')
        self.fam.on_point_undone('cp-UNKNOWN')
        self.assertEqual(len(self.fam.model._log), 1)   # entry still there

    def test_on_point_undone_removes_most_recent_matching_entry(self):
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [1, 2, 3], 'cp-1', 'm-1')
        self._add_point_bypassing_mirror(self.fam, 'seg-1', [4, 5, 6], 'cp-2', 'm-2')
        self.fam.on_point_undone('cp-1')
        remaining = [e for e in self.fam.model._log if e.get('type') == 'point']
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]['cp_id'], 'cp-2')

    # --- export_data multiple entries ---

    def test_export_data_multiple_points_all_returned(self):
        for i in range(3):
            self._add_point_bypassing_mirror(
                self.fam, 'seg-1', [float(i), 0.0, 0.0], f'cp-{i}', f'm-{i}'
            )
        points = self.fam.export_data()['segments']['seg-1']['points']
        self.assertEqual(len(points), 3)

    def test_export_data_preserves_order(self):
        for i in range(3):
            self._add_point_bypassing_mirror(
                self.fam, 'seg-1', [float(i), 0.0, 0.0], f'cp-{i}', f'm-{i}'
            )
        points = self.fam.export_data()['segments']['seg-1']['points']
        coords = [p['coord_ras'][0][0] for p in points.values()]
        self.assertEqual(coords, [0.0, 1.0, 2.0])

    # --- sync_visibility ---

    def _make_stub_scene_node(self, present=True):
        """Return a stub (scene, node) pair where node.SetDisplayVisibility is tracked."""
        class _StubNode:
            def __init__(self):
                self.visibility_calls = []

            def SetDisplayVisibility(self, v):
                self.visibility_calls.append(v)

        class _StubScene:
            def __init__(self, node, present):
                self._node = node
                self._present = present

            def IsNodePresent(self, node):
                return self._present if node is self._node else False

        node = _StubNode()
        scene = _StubScene(node, present)
        return scene, node

    def test_sync_visibility_current_seg_uses_current_visible(self):
        import sys
        scene, node = self._make_stub_scene_node(present=True)
        self.fam.model._nodes['seg-1'] = node
        # Patch slicer so _mirror_to_node's import resolves
        import types
        fake_slicer = types.ModuleType('slicer')
        fake_slicer.mrmlScene = scene
        sys.modules['slicer'] = fake_slicer
        try:
            self.fam.sync_visibility('seg-1', current_visible=True, saved_visible=False)
        finally:
            del sys.modules['slicer']
        self.assertEqual(node.visibility_calls, [1])

    def test_sync_visibility_saved_seg_uses_saved_visible(self):
        import sys, types
        scene, node = self._make_stub_scene_node(present=True)
        self.fam.model._nodes['seg-2'] = node
        fake_slicer = types.ModuleType('slicer')
        fake_slicer.mrmlScene = scene
        sys.modules['slicer'] = fake_slicer
        try:
            self.fam.sync_visibility('seg-1', current_visible=True, saved_visible=False)
        finally:
            del sys.modules['slicer']
        self.assertEqual(node.visibility_calls, [0])

    def test_sync_visibility_skips_absent_nodes(self):
        import sys, types
        scene, node = self._make_stub_scene_node(present=False)
        self.fam.model._nodes['seg-1'] = node
        fake_slicer = types.ModuleType('slicer')
        fake_slicer.mrmlScene = scene
        sys.modules['slicer'] = fake_slicer
        try:
            self.fam.sync_visibility('seg-1', current_visible=True, saved_visible=False)
        finally:
            del sys.modules['slicer']
        self.assertEqual(node.visibility_calls, [])

    def test_sync_visibility_multiple_segments(self):
        import sys, types

        class _MultiScene:
            def IsNodePresent(self, node):
                return True

        class _StubNode:
            def __init__(self):
                self.calls = []
            def SetDisplayVisibility(self, v):
                self.calls.append(v)

        scene = _MultiScene()
        node_a = _StubNode()
        node_b = _StubNode()
        self.fam.model._nodes['seg-a'] = node_a
        self.fam.model._nodes['seg-b'] = node_b

        fake_slicer = types.ModuleType('slicer')
        fake_slicer.mrmlScene = scene
        sys.modules['slicer'] = fake_slicer
        try:
            self.fam.sync_visibility('seg-a', current_visible=True, saved_visible=False)
        finally:
            del sys.modules['slicer']
        self.assertEqual(node_a.calls, [1])   # current → visible
        self.assertEqual(node_b.calls, [0])   # saved   → hidden

    # --- is_negative filtering ---

    def test_negative_point_not_logged(self):
        """on_point_confirmed with is_negative=True must not add a log entry."""
        # Patch _mirror_to_node so it doesn't call slicer
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'm-1'
        self.fam.on_point_confirmed('seg-1', [1, 2, 3], 'cp-1', is_negative=True)
        point_entries = [e for e in self.fam.model._log if e.get('type') == 'point']
        self.assertEqual(len(point_entries), 0)

    def test_positive_point_is_logged(self):
        """on_point_confirmed with is_negative=False (default) appends a log entry."""
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'm-1'
        self.fam.on_point_confirmed('seg-1', [1, 2, 3], 'cp-1', is_negative=False)
        point_entries = [e for e in self.fam.model._log if e.get('type') == 'point']
        self.assertEqual(len(point_entries), 1)

    def test_positive_point_default_is_not_negative(self):
        """on_point_confirmed omitting is_negative treats the point as positive."""
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'm-1'
        self.fam.on_point_confirmed('seg-1', [1, 2, 3], 'cp-1')
        point_entries = [e for e in self.fam.model._log if e.get('type') == 'point']
        self.assertEqual(len(point_entries), 1)

    def test_family_delegate_passes_is_negative(self):
        """TimedAnnotatorFamily.on_point_confirmed forwards is_negative to the model."""
        calls = []
        self.fam.model.on_point_confirmed = lambda *a, **kw: calls.append((a, kw))
        self.fam.on_point_confirmed('seg-1', [1, 2, 3], 'cp-1', is_negative=True)
        self.assertEqual(len(calls), 1)
        args, _ = calls[0]
        # is_negative is the 4th positional arg forwarded by the family delegate
        self.assertTrue(args[3])

    # --- load_from_json ---

    # nested format (new)

    def test_load_from_json_nested_adds_entries_to_log(self):
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'mirror-1'
        data = {'segments': {
            'seg-1': {'seg_name': 'Segment_1', 'points': [
                {'coord_ras': [1.0, 2.0, 3.0], 'timestamp': '2026-01-01T00:00:00'},
                {'coord_ras': [4.0, 5.0, 6.0], 'timestamp': '2026-01-01T00:00:01'},
            ]},
        }}
        n = self.fam.model.load_from_json(data)
        self.assertEqual(n, 2)
        point_entries = [e for e in self.fam.model._log if e.get('type') == 'point']
        self.assertEqual(len(point_entries), 2)

    def test_load_from_json_nested_two_segments(self):
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'm-1'
        data = {'segments': {
            'seg-1': {'seg_name': 'S1', 'points': [{'coord_ras': [0, 0, 0], 'timestamp': ''}]},
            'seg-2': {'seg_name': 'S2', 'points': [{'coord_ras': [1, 1, 1], 'timestamp': ''}]},
        }}
        n = self.fam.model.load_from_json(data)
        self.assertEqual(n, 2)

    def test_load_from_json_nested_preserves_coord_ras(self):
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'm-1'
        data = {'segments': {'seg-1': {'seg_name': 'S1', 'points': [
            {'coord_ras': [7.0, 8.0, 9.0], 'timestamp': ''}
        ]}}}
        self.fam.model.load_from_json(data)
        entry = next(e for e in self.fam.model._log if e.get('type') == 'point')
        self.assertEqual(entry['coord_ras'], [7.0, 8.0, 9.0])

    def test_load_from_json_nested_empty_segments_returns_zero(self):
        n = self.fam.model.load_from_json({'segments': {}})
        self.assertEqual(n, 0)

    # flat format (legacy backward-compat)

    def test_load_from_json_flat_adds_entries_to_log(self):
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'mirror-1'
        data = [
            {'segment_id': 'seg-1', 'coord_ras': [1.0, 2.0, 3.0], 'timestamp': '2026-01-01T00:00:00'},
            {'segment_id': 'seg-1', 'coord_ras': [4.0, 5.0, 6.0], 'timestamp': '2026-01-01T00:00:01'},
        ]
        n = self.fam.model.load_from_json(data)
        self.assertEqual(n, 2)
        point_entries = [e for e in self.fam.model._log if e.get('type') == 'point']
        self.assertEqual(len(point_entries), 2)

    def test_load_from_json_flat_returns_count(self):
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'm-1'
        data = [{'segment_id': 'seg-1', 'coord_ras': [0, 0, 0], 'timestamp': ''}]
        n = self.fam.model.load_from_json(data)
        self.assertEqual(n, 1)

    def test_load_from_json_flat_skips_entries_without_segment_id(self):
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'm-1'
        data = [{'coord_ras': [1.0, 2.0, 3.0], 'timestamp': '2026-01-01T00:00:00'}]
        n = self.fam.model.load_from_json(data)
        self.assertEqual(n, 0)

    def test_load_from_json_flat_preserves_coord_ras(self):
        self.fam.model._mirror_to_node = lambda seg_id, ras: 'm-1'
        data = [{'segment_id': 'seg-1', 'coord_ras': [7.0, 8.0, 9.0], 'timestamp': '2026-01-01T00:00:00'}]
        self.fam.model.load_from_json(data)
        entry = next(e for e in self.fam.model._log if e.get('type') == 'point')
        self.assertEqual(entry['coord_ras'], [7.0, 8.0, 9.0])

    def test_load_from_json_flat_empty_list_returns_zero(self):
        n = self.fam.model.load_from_json([])
        self.assertEqual(n, 0)

    # --- per-segment colors ---

    def test_first_segment_gets_first_palette_color(self):
        model = self.fam.model
        color = model._color_for('seg-1')
        self.assertEqual(color, model._PALETTE[0])

    def test_second_segment_gets_second_palette_color(self):
        model = self.fam.model
        model._color_for('seg-1')
        color = model._color_for('seg-2')
        self.assertEqual(color, model._PALETTE[1])

    def test_same_segment_always_returns_same_color(self):
        model = self.fam.model
        c1 = model._color_for('seg-1')
        c2 = model._color_for('seg-1')
        self.assertIs(c1, c2)

    def test_palette_cycles_after_exhaustion(self):
        model = self.fam.model
        palette_len = len(model._PALETTE)
        for i in range(palette_len):
            model._color_for(f'seg-{i}')
        # Next segment wraps to palette[0]
        color = model._color_for(f'seg-{palette_len}')
        self.assertEqual(color, model._PALETTE[0])

    def test_color_tuple_has_three_components(self):
        model = self.fam.model
        color = model._color_for('seg-x')
        self.assertEqual(len(color), 3)

    def test_load_from_json_reuses_existing_color_for_known_segment(self):
        model = self.fam.model
        model._mirror_to_node = lambda seg_id, ras: 'm-1'
        first_color = model._color_for('seg-1')
        data = {'segments': {'seg-1': {'seg_name': 'S1', 'points': [
            {'coord_ras': [0, 0, 0], 'timestamp': ''}
        ]}}}
        model.load_from_json(data)
        self.assertEqual(model._seg_colors['seg-1'], first_color)

    # --- visible buttons ---

    def test_visible_buttons_contains_import(self):
        self.assertIn('importAnnotationLogButton', TimedAnnotatorFamily.VISIBLE_BUTTONS)

    # --- registry ---

    def test_annotation_log_family_in_registry(self):
        from core.modelFamilies import FAMILY_REGISTRY
        self.assertIn('TimedMarker', FAMILY_REGISTRY)
        self.assertIs(FAMILY_REGISTRY['TimedMarker'], TimedAnnotatorFamily)


if __name__ == '__main__':
    unittest.main()
