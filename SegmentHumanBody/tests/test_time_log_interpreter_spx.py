"""Tests for TimeLogInterpreter handling of SPX / fill-hole / model events."""
import unittest
from core.TimeLogInterpreter import TimeLogInterpreter

RAW_TYPE = 'SegmentHumanBody.raw_input'


def _make_raw(events):
    return {
        'type': RAW_TYPE,
        'metadata': {},
        'events': events,
    }


def _interp(events):
    return TimeLogInterpreter(_make_raw(events)).export()['events']


class TestInterpreterSPXEvents(unittest.TestCase):

    def test_spx_brush_fill_passes_through(self):
        events = _interp([{
            'event': 'spx_brush_fill',
            'timestamp': '2025-01-01T00:00:00.000',
            'label_id': 3,
            'delta_pixels': 42,
            'view': 'Red',
            'axis': 'ax',
            'slice_idx': 5,
            'model_key': 'SPX_Tester2D',
            'params': {'gh': 9},
            'segment_id': 'seg-1',
            'segmentation_id': 'sn-1',
            'volume_id': 'vol-1',
            'operation': 'spx_expand',
            'additive': True,
        }])
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev['event'], 'spx_brush_fill')
        self.assertEqual(ev['label_id'], 3)
        self.assertEqual(ev['delta_pixels'], 42)
        self.assertEqual(ev['operation'], 'spx_expand')
        self.assertTrue(ev['additive'])

    def test_spx_erase_fill_passes_through(self):
        events = _interp([{
            'event': 'spx_erase_fill',
            'timestamp': '2025-01-01T00:00:01.000',
            'label_id': 7,
            'delta_pixels': 10,
            'view': 'Red',
            'axis': 'ax',
            'slice_idx': 5,
            'model_key': 'SPX_Tester2D',
            'params': {},
            'segment_id': 'seg-1',
            'segmentation_id': 'sn-1',
            'volume_id': 'vol-1',
            'operation': 'spx_expand',
            'additive': False,
        }])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event'], 'spx_erase_fill')
        self.assertEqual(events[0]['label_id'], 7)

    def test_fill_hole_passes_through(self):
        events = _interp([{
            'event': 'fill_hole',
            'timestamp': '2025-01-01T00:00:02.000',
            'view': 'Green',
            'axis': 'cor',
            'slice_idx': 10,
            'delta_pixels': 200,
            'segment_id': 'seg-1',
            'segmentation_id': 'sn-1',
            'volume_id': 'vol-1',
        }])
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev['event'], 'fill_hole')
        self.assertEqual(ev['slice_idx'], 10)
        self.assertEqual(ev['delta_pixels'], 200)

    def test_spx_boundary_on_maps_to_on(self):
        events = _interp([{
            'event': 'spx_boundary_toggled',
            'timestamp': '2025-01-01T00:00:03.000',
            'visible': True,
            'view': 'Red',
            'slice_idx': 5,
            'model_key': 'SPX_Tester2D',
        }])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event'], 'spx_boundary_on')

    def test_spx_boundary_off_maps_to_off(self):
        events = _interp([{
            'event': 'spx_boundary_toggled',
            'timestamp': '2025-01-01T00:00:04.000',
            'visible': False,
            'view': 'Red',
            'slice_idx': 5,
            'model_key': 'SPX_Tester2D',
        }])
        self.assertEqual(events[0]['event'], 'spx_boundary_off')

    def test_model_family_changed_passes_through(self):
        events = _interp([{
            'event': 'model_family_changed',
            'timestamp': '2025-01-01T00:00:05.000',
            'family': 'SPX-Assisted Annotation',
        }])
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev['event'], 'model_family_changed')
        self.assertEqual(ev['family'], 'SPX-Assisted Annotation')

    def test_model_variant_changed_passes_through(self):
        events = _interp([{
            'event': 'model_variant_changed',
            'timestamp': '2025-01-01T00:00:06.000',
            'variant': 'Naive_Grid-2D',
        }])
        self.assertEqual(events[0]['event'], 'model_variant_changed')
        self.assertEqual(events[0]['variant'], 'Naive_Grid-2D')

    def test_model_confirmed_passes_through(self):
        events = _interp([{
            'event': 'model_confirmed',
            'timestamp': '2025-01-01T00:00:07.000',
            'family': 'SPX-Assisted Annotation',
            'variant': 'Naive_Grid-2D',
        }])
        ev = events[0]
        self.assertEqual(ev['event'], 'model_confirmed')
        self.assertEqual(ev['family'], 'SPX-Assisted Annotation')
        self.assertEqual(ev['variant'], 'Naive_Grid-2D')

    # ------------------------------------------------------------------ #
    # Overwrite mode                                                      #
    # ------------------------------------------------------------------ #

    def test_overwrite_mode_changed_produces_compact_event(self):
        events = _interp([{
            'event': 'overwrite_mode_changed',
            'timestamp': '2025-01-01T00:00:08.000',
            'mode': 'OverwriteAllSegments',
            'mode_label': 'Aggressive — overwrite all',
        }])
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev['event'], 'overwrite_mode_change')
        self.assertEqual(ev['mode'], 'OverwriteAllSegments')
        self.assertEqual(ev['mode_label'], 'Aggressive — overwrite all')
        self.assertIn('timestamp', ev)

    def test_overwrite_mode_coexist(self):
        events = _interp([{
            'event': 'overwrite_mode_changed',
            'timestamp': '2025-01-01T00:00:09.000',
            'mode': 'OverwriteNone',
            'mode_label': 'Coexist — never erase others',
        }])
        ev = events[0]
        self.assertEqual(ev['mode'], 'OverwriteNone')
        self.assertEqual(ev['mode_label'], 'Coexist — never erase others')

    def test_overwrite_mode_defensive(self):
        events = _interp([{
            'event': 'overwrite_mode_changed',
            'timestamp': '2025-01-01T00:00:10.000',
            'mode': 'PaintAllowedOutsideAllSegments',
            'mode_label': 'Defensive — blank areas only',
        }])
        ev = events[0]
        self.assertEqual(ev['mode'], 'PaintAllowedOutsideAllSegments')
        self.assertEqual(ev['mode_label'], 'Defensive — blank areas only')

    def test_initial_overwrite_mode_in_compact_metadata_when_present(self):
        raw = {
            'type': RAW_TYPE,
            'metadata': {'initial_overwrite_mode': {
                'mode': 'OverwriteNone',
                'mode_label': 'Coexist — never erase others',
            }},
            'events': [],
        }
        out = TimeLogInterpreter(raw).export()
        om = out['metadata']['initial_overwrite_mode']
        self.assertEqual(om['mode'], 'OverwriteNone')
        self.assertEqual(om['mode_label'], 'Coexist — never erase others')

    def test_initial_overwrite_mode_defaults_to_overwrite_none_for_old_logs(self):
        """Old raw logs without initial_overwrite_mode default to OverwriteNone (Coexist)."""
        raw = {'type': RAW_TYPE, 'metadata': {}, 'events': []}
        out = TimeLogInterpreter(raw).export()
        om = out['metadata']['initial_overwrite_mode']
        self.assertEqual(om['mode'], 'OverwriteNone')

    def test_ids_are_sequential(self):
        events = _interp([
            {'event': 'spx_brush_fill', 'timestamp': 't1', 'label_id': 1,
             'delta_pixels': 5, 'view': 'Red', 'axis': 'ax', 'slice_idx': 0,
             'model_key': 'k', 'params': {}, 'segment_id': 's',
             'segmentation_id': 'sn', 'volume_id': 'v',
             'operation': 'spx_expand', 'additive': True},
            {'event': 'fill_hole', 'timestamp': 't2',
             'view': 'Red', 'axis': 'ax', 'slice_idx': 0, 'delta_pixels': 10,
             'segment_id': 's', 'segmentation_id': 'sn', 'volume_id': 'v'},
        ])
        ids = [e['id'] for e in events]
        self.assertEqual(ids, list(range(1, len(ids) + 1)))
