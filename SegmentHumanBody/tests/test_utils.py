import unittest
import numpy as np

from core.utils import (
    call_if_exists,
    get_slice_from_volume,
    write_slice_to_volume,
    extract_connected_component,
)


class TestCallIfExists(unittest.TestCase):

    def test_calls_existing_method_and_returns_value(self):
        class Obj:
            def greet(self):
                return 42

        self.assertEqual(call_if_exists(Obj(), 'greet'), 42)

    def test_returns_none_when_method_absent(self):
        class Obj:
            pass

        self.assertIsNone(call_if_exists(Obj(), 'missing'))

    def test_returns_none_when_obj_is_none(self):
        self.assertIsNone(call_if_exists(None, 'anything'))

    def test_passes_args_and_kwargs(self):
        class Obj:
            def add(self, a, b=0):
                return a + b

        self.assertEqual(call_if_exists(Obj(), 'add', 3, b=4), 7)


class TestGetSliceFromVolume(unittest.TestCase):

    def setUp(self):
        # shape (Z=4, Y=5, X=6)
        self.vol = np.arange(4 * 5 * 6).reshape(4, 5, 6)

    def test_axis0_returns_correct_slice(self):
        result = get_slice_from_volume(self.vol, axis=0, slice_index=2)
        np.testing.assert_array_equal(result, self.vol[2, :, :])

    def test_axis1_returns_correct_slice(self):
        result = get_slice_from_volume(self.vol, axis=1, slice_index=3)
        np.testing.assert_array_equal(result, self.vol[:, 3, :])

    def test_axis2_returns_correct_slice(self):
        result = get_slice_from_volume(self.vol, axis=2, slice_index=1)
        np.testing.assert_array_equal(result, self.vol[:, :, 1])

    def test_returned_slice_is_view(self):
        """Slices are numpy views; mutations propagate to the original."""
        s = get_slice_from_volume(self.vol, axis=0, slice_index=0)
        s[0, 0] = -999
        self.assertEqual(self.vol[0, 0, 0], -999)


class TestWriteSliceToVolume(unittest.TestCase):

    def setUp(self):
        self.vol = np.zeros((4, 5, 6), dtype=np.int32)

    def _make_slice(self, shape, fill=1):
        return np.full(shape, fill, dtype=np.int32)

    def test_axis0_writes_correctly(self):
        s = self._make_slice((5, 6), fill=7)
        write_slice_to_volume(self.vol, s, axis=0, slice_index=1)
        np.testing.assert_array_equal(self.vol[1, :, :], s)
        self.assertTrue(np.all(self.vol[0, :, :] == 0))

    def test_axis1_writes_correctly(self):
        s = self._make_slice((4, 6), fill=3)
        write_slice_to_volume(self.vol, s, axis=1, slice_index=2)
        np.testing.assert_array_equal(self.vol[:, 2, :], s)

    def test_axis2_writes_correctly(self):
        s = self._make_slice((4, 5), fill=5)
        write_slice_to_volume(self.vol, s, axis=2, slice_index=4)
        np.testing.assert_array_equal(self.vol[:, :, 4], s)

    def test_roundtrip_get_write(self):
        original = np.random.randint(0, 255, (4, 5, 6), dtype=np.int32)
        target = np.zeros_like(original)
        for i in range(4):
            s = get_slice_from_volume(original, axis=0, slice_index=i)
            write_slice_to_volume(target, s, axis=0, slice_index=i)
        np.testing.assert_array_equal(target, original)


class TestExtractConnectedComponent(unittest.TestCase):

    def test_seed_inside_region_returns_component(self):
        mask = np.array([
            [True,  True,  False],
            [True,  False, False],
            [False, False, True ],
        ])
        result = extract_connected_component(mask, point_xy=(0, 0))  # x=0,y=0
        # Top-left connected region: (0,0),(1,0),(0,1)
        self.assertTrue(result[0, 0])
        self.assertTrue(result[1, 0])
        self.assertTrue(result[0, 1])
        # Isolated True at bottom-right must NOT be included
        self.assertFalse(result[2, 2])

    def test_seed_on_false_pixel_returns_empty(self):
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        result = extract_connected_component(mask, point_xy=(1, 1))
        self.assertFalse(result.any())

    def test_all_true_mask_returns_full_component(self):
        mask = np.ones((4, 4), dtype=bool)
        result = extract_connected_component(mask, point_xy=(2, 2))
        np.testing.assert_array_equal(result, mask)


if __name__ == '__main__':
    unittest.main()
