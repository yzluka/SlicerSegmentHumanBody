"""Tests for TimeLogSummarizer handling of SPX / fill-hole / model spans."""
import unittest
from core.TimeLogSummarizer import TimeLogSummarizer

INTERP_TYPE = 'SegmentHumanBody.annotation_process'


def _make_log(events):
    for i, ev in enumerate(events, 1):
        ev.setdefault('id', i)
        ev.setdefault('timestamp', f'2025-01-01T00:00:{i:02d}.000')
    return {
        'type': INTERP_TYPE,
        'metadata': {},
        'events': events,
    }


def _spans(events):
    return TimeLogSummarizer(_make_log(events)).spans()


class TestSummariserSPXStrokes(unittest.TestCase):

    def test_consecutive_spx_brush_fills_same_slice_merge(self):
        evs = [
            {'event': 'spx_brush_fill', 'label_id': 3, 'delta_pixels': 20,
             'segment_id': 'seg-1', 'slice_idx': 5, 'view': 'Red', 'model_key': 'k'},
            {'event': 'spx_brush_fill', 'label_id': 5, 'delta_pixels': 15,
             'segment_id': 'seg-1', 'slice_idx': 5, 'view': 'Red', 'model_key': 'k'},
            {'event': 'spx_brush_fill', 'label_id': 7, 'delta_pixels': 10,
             'segment_id': 'seg-1', 'slice_idx': 5, 'view': 'Red', 'model_key': 'k'},
        ]
        spans = _spans(evs)
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span['type'], 'spx_brush_stroke')
        self.assertEqual(span['total_delta_pixels'], 45)
        self.assertIn(3, span['labels'])
        self.assertIn(5, span['labels'])
        self.assertIn(7, span['labels'])

    def test_spx_brush_fills_different_slice_do_not_merge(self):
        evs = [
            {'event': 'spx_brush_fill', 'label_id': 1, 'delta_pixels': 10,
             'segment_id': 'seg-1', 'slice_idx': 5, 'view': 'Red', 'model_key': 'k'},
            {'event': 'spx_brush_fill', 'label_id': 2, 'delta_pixels': 10,
             'segment_id': 'seg-1', 'slice_idx': 6, 'view': 'Red', 'model_key': 'k'},
        ]
        spans = _spans(evs)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]['type'], 'spx_brush_stroke')
        self.assertEqual(spans[1]['type'], 'spx_brush_stroke')

    def test_spx_erase_fills_merge_into_erase_stroke(self):
        evs = [
            {'event': 'spx_erase_fill', 'label_id': 4, 'delta_pixels': 8,
             'segment_id': 'seg-1', 'slice_idx': 3, 'view': 'Red', 'model_key': 'k'},
            {'event': 'spx_erase_fill', 'label_id': 6, 'delta_pixels': 12,
             'segment_id': 'seg-1', 'slice_idx': 3, 'view': 'Red', 'model_key': 'k'},
        ]
        spans = _spans(evs)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['type'], 'spx_erase_stroke')
        self.assertEqual(spans[0]['total_delta_pixels'], 20)

    def test_fill_hole_produces_single_span(self):
        evs = [
            {'event': 'fill_hole', 'delta_pixels': 200,
             'segment_id': 'seg-1', 'slice_idx': 7, 'view': 'Green'},
        ]
        spans = _spans(evs)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['type'], 'fill_hole')
        self.assertEqual(spans[0]['delta_pixels'], 200)

    def test_spx_boundary_on_and_off_merge(self):
        evs = [
            {'event': 'spx_boundary_on', 'view': 'Red', 'model_key': 'k'},
            {'event': 'spx_boundary_off', 'view': 'Red', 'model_key': 'k'},
        ]
        spans = _spans(evs)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['type'], 'spx_boundary_inspection')

    def test_spx_boundary_on_without_off_single_span(self):
        evs = [
            {'event': 'spx_boundary_on', 'view': 'Yellow', 'model_key': 'k'},
        ]
        spans = _spans(evs)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['type'], 'spx_boundary_inspection')

    def test_model_confirmed_produces_model_change_span(self):
        evs = [
            {'event': 'model_confirmed', 'family': 'SPX-Assisted Annotation',
             'variant': 'Naive_Grid-2D'},
        ]
        spans = _spans(evs)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['type'], 'model_change')
        self.assertEqual(spans[0]['family'], 'SPX-Assisted Annotation')

    def test_model_family_changed_produces_model_change_span(self):
        evs = [{'event': 'model_family_changed', 'family': 'Default'}]
        spans = _spans(evs)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['type'], 'model_change')

    def test_multiple_span_types_interleaved(self):
        evs = [
            {'event': 'spx_brush_fill', 'label_id': 1, 'delta_pixels': 5,
             'segment_id': 'seg-1', 'slice_idx': 5, 'view': 'Red', 'model_key': 'k'},
            {'event': 'fill_hole', 'delta_pixels': 100,
             'segment_id': 'seg-1', 'slice_idx': 5, 'view': 'Red'},
            {'event': 'model_confirmed', 'family': 'Auto', 'variant': 'BreastCT'},
        ]
        spans = _spans(evs)
        types = [s['type'] for s in spans]
        self.assertIn('spx_brush_stroke', types)
        self.assertIn('fill_hole', types)
        self.assertIn('model_change', types)
