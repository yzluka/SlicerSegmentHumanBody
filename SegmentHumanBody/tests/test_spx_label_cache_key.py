"""Tests for MRML-anchored SPX label cache key behaviour."""
import unittest
import numpy as np
from core.modelFamilies import SPXModelFamily


class _FakeVolumeNode:
    def __init__(self, node_id='vol-1', mtime=100):
        self._id = node_id
        self._mtime = mtime

    def GetID(self):
        return self._id

    def GetMTime(self):
        return self._mtime

    def bump_mtime(self):
        self._mtime += 1


class _CountingModel:
    def __init__(self):
        self.call_count = 0

    def forward(self, **kwargs):
        self.call_count += 1
        img = kwargs.get('img')
        return np.zeros(img.shape[:2], dtype=np.int32)


class TestMRMLKeyedCache(unittest.TestCase):

    def _make_fam(self):
        fam = SPXModelFamily(variant='Naive_Grid-2D')
        fam.model = _CountingModel()
        return fam

    def test_cache_hit_on_same_mrml_identity(self):
        fam = self._make_fam()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        self.assertEqual(fam.model.call_count, 1)

    def test_cache_miss_on_different_slice_idx(self):
        fam = self._make_fam()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=6, img=img)
        self.assertEqual(fam.model.call_count, 2)

    def test_cache_stable_on_volume_mtime_bump(self):
        """MTime bump must NOT cause a cache miss.
        Slicer increments volume MTime after each modifySelectedSegmentByLabelmap
        (rendering update), so MTime is intentionally excluded from the cache key."""
        fam = self._make_fam()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        vol.bump_mtime()
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        self.assertEqual(fam.model.call_count, 1)

    def test_cache_miss_on_different_volume_node(self):
        fam = self._make_fam()
        vol1 = _FakeVolumeNode(node_id='vol-1')
        vol2 = _FakeVolumeNode(node_id='vol-2')
        img = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol1, axis='ax', slice_idx=5, img=img)
        fam.on_expand(volume_node=vol2, axis='ax', slice_idx=5, img=img)
        self.assertEqual(fam.model.call_count, 2)

    def test_cache_miss_on_different_axis(self):
        fam = self._make_fam()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img)
        fam.on_expand(volume_node=vol, axis='cor', slice_idx=5, img=img)
        self.assertEqual(fam.model.call_count, 2)

    def test_cache_miss_on_different_params(self):
        fam = self._make_fam()
        vol = _FakeVolumeNode()
        img = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img, gh=9)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img, gh=12)
        self.assertEqual(fam.model.call_count, 2)

    def test_cache_survives_buffer_reallocation(self):
        """Cache hit even when the numpy array is a different object in memory."""
        fam = self._make_fam()
        vol = _FakeVolumeNode()
        img1 = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img1)
        # Create a brand-new array with the same data — old key was pointer-based
        # but new key is MRML-based, so still a cache hit.
        img2 = np.zeros((10, 20), dtype=np.uint8)
        fam.on_expand(volume_node=vol, axis='ax', slice_idx=5, img=img2)
        self.assertEqual(fam.model.call_count, 1)

    def test_unhashable_kwargs_fallback(self):
        fam = self._make_fam()
        vol = _FakeVolumeNode()
        key = fam._make_cache_key(vol, 'ax', 0, {'weights': [1, 2, 3]})
        self.assertIsNotNone(key)
