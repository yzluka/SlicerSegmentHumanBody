"""Tests for core/_point_log.py — pure-Python per-segment control-point store."""
from core._point_log import PointLog


def _entry(cp_id, is_neg=False, ras=None):
    return {'ras': ras or [0.0, 0.0, 0.0], 'is_neg': is_neg, 'cp_id': cp_id}


# ── append / get ──────────────────────────────────────────────────────────────

def test_append_adds_entry():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    assert len(log.get('seg-a')) == 1


def test_append_preserves_insertion_order():
    log = PointLog()
    for i in range(5):
        log.append('seg-a', _entry(f'cp{i}'))
    ids = [e['cp_id'] for e in log.get('seg-a')]
    assert ids == ['cp0', 'cp1', 'cp2', 'cp3', 'cp4']


def test_get_missing_segment_returns_empty_list():
    assert PointLog().get('seg-x') == []


def test_get_returns_shallow_copy():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    copy = log.get('seg-a')
    copy.append(_entry('extra'))
    assert len(log.get('seg-a')) == 1


def test_append_creates_independent_lists_per_segment():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    log.append('seg-b', _entry('cp2'))
    assert len(log.get('seg-a')) == 1
    assert len(log.get('seg-b')) == 1


# ── save ─────────────────────────────────────────────────────────────────────

def test_save_replaces_existing_entries():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    log.save('seg-a', [_entry('cp2'), _entry('cp3')])
    ids = [e['cp_id'] for e in log.get('seg-a')]
    assert ids == ['cp2', 'cp3']


def test_save_on_empty_segment_creates_entries():
    log = PointLog()
    log.save('seg-a', [_entry('cp1')])
    assert log.get('seg-a')[0]['cp_id'] == 'cp1'


def test_save_with_empty_list_clears_segment():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    log.save('seg-a', [])
    assert log.get('seg-a') == []


# ── sync_removed ─────────────────────────────────────────────────────────────

def test_sync_removed_drops_absent_positive_entries():
    log = PointLog()
    log.append('seg-a', _entry('cp1', is_neg=False))
    log.append('seg-a', _entry('cp2', is_neg=False))
    log.sync_removed('seg-a', is_neg=False, present_cp_ids={'cp2'})
    ids = [e['cp_id'] for e in log.get('seg-a')]
    assert ids == ['cp2']


def test_sync_removed_drops_absent_negative_entries():
    log = PointLog()
    log.append('seg-a', _entry('cp1', is_neg=True))
    log.append('seg-a', _entry('cp2', is_neg=True))
    log.sync_removed('seg-a', is_neg=True, present_cp_ids={'cp1'})
    ids = [e['cp_id'] for e in log.get('seg-a')]
    assert ids == ['cp1']


def test_sync_removed_preserves_opposite_polarity():
    log = PointLog()
    log.append('seg-a', _entry('pos', is_neg=False))
    log.append('seg-a', _entry('neg', is_neg=True))
    log.sync_removed('seg-a', is_neg=False, present_cp_ids=set())
    entries = log.get('seg-a')
    assert len(entries) == 1
    assert entries[0]['cp_id'] == 'neg'


def test_sync_removed_on_missing_segment_is_noop():
    log = PointLog()
    log.sync_removed('seg-x', is_neg=False, present_cp_ids={'cp1'})
    assert log.all_segments() == []


def test_sync_removed_on_empty_entries_is_noop():
    log = PointLog()
    log.save('seg-a', [])
    log.sync_removed('seg-a', is_neg=False, present_cp_ids=set())
    assert log.get('seg-a') == []


# ── remove_segment / clear ───────────────────────────────────────────────────

def test_remove_segment_deletes_all_entries_for_that_segment():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    log.append('seg-a', _entry('cp2'))
    log.append('seg-b', _entry('cp3'))
    log.remove_segment('seg-a')
    assert log.get('seg-a') == []
    assert len(log.get('seg-b')) == 1


def test_remove_segment_on_missing_id_is_noop():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    log.remove_segment('seg-x')
    assert len(log.get('seg-a')) == 1


def test_clear_removes_all_segments():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    log.append('seg-b', _entry('cp2'))
    log.clear()
    assert log.all_segments() == []
    assert log.get('seg-a') == []
    assert log.get('seg-b') == []


# ── all_segments ──────────────────────────────────────────────────────────────

def test_all_segments_returns_ids_that_have_entries():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    log.append('seg-b', _entry('cp2'))
    assert set(log.all_segments()) == {'seg-a', 'seg-b'}


def test_all_segments_includes_segment_saved_with_empty_list():
    log = PointLog()
    log.save('seg-a', [])
    assert 'seg-a' in log.all_segments()


# ── export ────────────────────────────────────────────────────────────────────

def test_export_returns_complete_data():
    log = PointLog()
    log.append('seg-a', _entry('cp1', ras=[1.0, 2.0, 3.0]))
    log.append('seg-b', _entry('cp2', is_neg=True))
    exported = log.export()
    assert set(exported.keys()) == {'seg-a', 'seg-b'}
    assert exported['seg-a'][0]['cp_id'] == 'cp1'


def test_export_is_deep_copy_not_aliased():
    log = PointLog()
    log.append('seg-a', _entry('cp1'))
    exported = log.export()
    exported['seg-a'][0]['cp_id'] = 'mutated'
    assert log.get('seg-a')[0]['cp_id'] == 'cp1'


def test_export_mutating_ras_does_not_affect_original():
    log = PointLog()
    log.append('seg-a', _entry('cp1', ras=[1.0, 2.0, 3.0]))
    exported = log.export()
    exported['seg-a'][0]['ras'][0] = 99.0
    assert log.get('seg-a')[0]['ras'][0] == 1.0
