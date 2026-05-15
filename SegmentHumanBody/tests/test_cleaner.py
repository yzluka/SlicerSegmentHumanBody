"""Tests for tools/audio_processor/cleaner.py."""
import json
import os
import sys
import warnings

import pytest

# Add the audio_processor package to sys.path (it lives outside SegmentHumanBody/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools', 'audio_processor'))
from cleaner import load_dir, clean_text, clean_phrases


# ── load_dir ──────────────────────────────────────────────────────────────────

def test_load_dir_reads_corrections_from_json_files(tmp_path):
    (tmp_path / 'anatomy.json').write_text(
        json.dumps({'femur': 'Femur', 'tibia': 'Tibia'}), encoding='utf-8')
    corrections, _ = load_dir(tmp_path)
    assert corrections == {'femur': 'Femur', 'tibia': 'Tibia'}


def test_load_dir_reads_review_patterns(tmp_path):
    (tmp_path / '_review_patterns.json').write_text(
        json.dumps(['check this', 'verify']), encoding='utf-8')
    _, patterns = load_dir(tmp_path)
    assert patterns == ['check this', 'verify']


def test_load_dir_merges_multiple_correction_files(tmp_path):
    (tmp_path / 'anatomy.json').write_text(
        json.dumps({'femur': 'Femur'}), encoding='utf-8')
    (tmp_path / 'actions.json').write_text(
        json.dumps({'brush': 'Brush'}), encoding='utf-8')
    corrections, _ = load_dir(tmp_path)
    assert 'femur' in corrections
    assert 'brush' in corrections


def test_load_dir_duplicate_key_emits_warning(tmp_path):
    (tmp_path / 'a.json').write_text(json.dumps({'key': 'v1'}), encoding='utf-8')
    (tmp_path / 'b.json').write_text(json.dumps({'key': 'v2'}), encoding='utf-8')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        corrections, _ = load_dir(tmp_path)
    assert any('duplicate key' in str(warning.message).lower() for warning in w)
    assert corrections['key'] == 'v2'  # last file wins


def test_load_dir_skips_invalid_json_with_warning(tmp_path):
    (tmp_path / 'bad.json').write_text('not valid json', encoding='utf-8')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        corrections, _ = load_dir(tmp_path)
    assert any('bad.json' in str(warning.message) for warning in w)
    assert corrections == {}


def test_load_dir_warns_when_review_file_is_not_array(tmp_path):
    (tmp_path / '_review_patterns.json').write_text(
        json.dumps({'not': 'an array'}), encoding='utf-8')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _, patterns = load_dir(tmp_path)
    assert any('_review_patterns.json' in str(warning.message) for warning in w)
    assert patterns == []


def test_load_dir_warns_when_correction_file_is_not_dict(tmp_path):
    (tmp_path / 'bad_format.json').write_text(
        json.dumps(['not', 'a', 'dict']), encoding='utf-8')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        corrections, _ = load_dir(tmp_path)
    assert any('bad_format.json' in str(warning.message) for warning in w)
    assert corrections == {}


def test_load_dir_empty_directory_returns_empty():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        corrections, patterns = load_dir(d)
    assert corrections == {}
    assert patterns == []


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_applies_correction():
    cleaned, applied, _ = clean_text('i see a femur', {'femur': 'Femur'}, [])
    assert cleaned == 'i see a Femur'
    assert len(applied) == 1
    assert applied[0]['pattern'] == 'femur'


def test_clean_text_is_case_insensitive():
    cleaned, _, _ = clean_text('FEMUR here', {'femur': 'Femur'}, [])
    assert cleaned == 'Femur here'


def test_clean_text_applies_multiple_corrections_in_order():
    corrections = {'femur': 'Femur', 'bone': 'Bone'}
    cleaned, applied, _ = clean_text('femur bone', corrections, [])
    assert 'Femur' in cleaned
    assert 'Bone' in cleaned
    assert len(applied) == 2


def test_clean_text_unchanged_text_returns_empty_applied():
    _, applied, _ = clean_text('unchanged', {'femur': 'Femur'}, [])
    assert applied == []


def test_clean_text_normalizes_whitespace():
    cleaned, _, _ = clean_text('too   many   spaces', {}, [])
    assert cleaned == 'too many spaces'


def test_clean_text_removes_space_before_punctuation():
    cleaned, _, _ = clean_text('hello , world .', {}, [])
    assert cleaned == 'hello, world.'


def test_clean_text_detects_review_pattern():
    _, _, needs_review = clean_text('check this phrase', {}, ['check this'])
    assert needs_review is True


def test_clean_text_no_review_when_no_pattern_matches():
    _, _, needs_review = clean_text('clean phrase', {}, ['check this'])
    assert needs_review is False


def test_clean_text_review_pattern_applied_after_corrections():
    corrections = {'badword': 'goodword'}
    review_patterns = ['goodword']
    text = 'badword present'
    cleaned, _, needs_review = clean_text(text, corrections, review_patterns)
    assert cleaned == 'goodword present'
    assert needs_review is True


def test_clean_text_empty_text_returns_empty():
    cleaned, applied, needs_review = clean_text('', {}, [])
    assert cleaned == ''
    assert applied == []
    assert needs_review is False


# ── clean_phrases ─────────────────────────────────────────────────────────────

def test_clean_phrases_returns_phrases_unchanged_when_no_dir():
    phrases = [{'text': 'femur here', 'start': 0.0}]
    result = clean_phrases(phrases, corrections_dir=None)
    assert result == phrases


def test_clean_phrases_adds_cleaned_text_when_changed(tmp_path):
    (tmp_path / 'anatomy.json').write_text(
        json.dumps({'femur': 'Femur'}), encoding='utf-8')
    phrases = [{'text': 'femur segment', 'start': 0.0}]
    result = clean_phrases(phrases, corrections_dir=tmp_path)
    assert result[0]['cleaned_text'] == 'Femur segment'


def test_clean_phrases_does_not_add_cleaned_text_when_unchanged(tmp_path):
    (tmp_path / 'anatomy.json').write_text(
        json.dumps({'bone': 'Bone'}), encoding='utf-8')
    phrases = [{'text': 'already clean', 'start': 0.0}]
    result = clean_phrases(phrases, corrections_dir=tmp_path)
    assert 'cleaned_text' not in result[0]


def test_clean_phrases_adds_corrections_list_when_applied(tmp_path):
    (tmp_path / 'anatomy.json').write_text(
        json.dumps({'femur': 'Femur'}), encoding='utf-8')
    phrases = [{'text': 'femur', 'start': 0.0}]
    result = clean_phrases(phrases, corrections_dir=tmp_path)
    assert 'corrections' in result[0]
    assert result[0]['corrections'][0]['pattern'] == 'femur'


def test_clean_phrases_adds_needs_review_flag(tmp_path):
    (tmp_path / '_review_patterns.json').write_text(
        json.dumps(['check']), encoding='utf-8')
    phrases = [{'text': 'please check this'}]
    result = clean_phrases(phrases, corrections_dir=tmp_path)
    assert result[0].get('needs_review') is True


def test_clean_phrases_preserves_original_fields(tmp_path):
    (tmp_path / 'a.json').write_text(json.dumps({'x': 'X'}), encoding='utf-8')
    phrase = {'text': 'x here', 'start': 1.5, 'end': 2.5, 'custom': 'data'}
    result = clean_phrases([phrase], corrections_dir=tmp_path)
    assert result[0]['start'] == 1.5
    assert result[0]['end'] == 2.5
    assert result[0]['custom'] == 'data'


def test_clean_phrases_processes_multiple_phrases(tmp_path):
    (tmp_path / 'a.json').write_text(
        json.dumps({'femur': 'Femur', 'tibia': 'Tibia'}), encoding='utf-8')
    phrases = [
        {'text': 'femur region'},
        {'text': 'tibia area'},
        {'text': 'no match here'},
    ]
    result = clean_phrases(phrases, corrections_dir=tmp_path)
    assert result[0]['cleaned_text'] == 'Femur region'
    assert result[1]['cleaned_text'] == 'Tibia area'
    assert 'cleaned_text' not in result[2]


def test_clean_phrases_does_not_mutate_input(tmp_path):
    (tmp_path / 'a.json').write_text(json.dumps({'x': 'X'}), encoding='utf-8')
    original = {'text': 'x here'}
    phrases = [original]
    clean_phrases(phrases, corrections_dir=tmp_path)
    assert original == {'text': 'x here'}
