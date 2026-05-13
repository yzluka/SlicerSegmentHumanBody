from core.TimeLogSummarizer import TimeLogSummarizer


def _log(events):
    return {
        'type': 'SegmentHumanBody.annotation_process',
        'metadata': {'volume': {'name': 'tissuerange.nii.gz'}},
        'events': events,
    }


def test_summarizer_groups_brush_stroke_with_context():
    events = [
        {
            'id': 1,
            'event': 'press',
            'timestamp': '2026-05-13T13:55:10.216',
            'ijk': [355, 226, 448],
            'view': 'Red',
            'tool': 'brush',
            'segment': 'Segment_1',
            'brush_mm': 19.0838,
        },
        {
            'id': 2,
            'event': 'hold',
            'timestamp': ['2026-05-13T13:55:10.278'],
            'ijk': [[352, 223, 448]],
            'view': 'Red',
        },
        {
            'id': 3,
            'event': 'release',
            'timestamp': '2026-05-13T13:55:12.421',
            'ijk': [357, 237, 448],
            'view': 'Red',
        },
    ]

    result = TimeLogSummarizer(_log(events)).export()
    text = result['text'][0]

    assert text.splitlines()[0] == (
        '(13:55:10–13:55:12: Volume=tissuerange.nii.gz, '
        'Segment=Segment_1, View=Red, Tool=brush, brush_mm=19.0838, '
        'slice=448, start=(355,226,448), end=(357,237,448))'
    )
    assert text.splitlines()[1] == (
        '  trajectory=[(355,226,448), (352,223,448), (357,237,448)]'
    )
    assert result['spans'][0]['trajectory'] == [
        [355, 226, 448], [352, 223, 448], [357, 237, 448],
    ]


def test_summarizer_groups_click_without_hold():
    events = [
        {
            'id': 1,
            'event': 'press',
            'timestamp': '2026-05-13T13:55:25.728',
            'ijk': [175, 108, 448],
            'view': 'Red',
            'tool': 'brush',
            'segment': 'Segment_1',
            'brush_mm': 19.0838,
        },
        {
            'id': 2,
            'event': 'release',
            'timestamp': '2026-05-13T13:55:25.837',
            'ijk': [175, 108, 448],
            'view': 'Red',
        },
    ]

    text = TimeLogSummarizer(_log(events)).export_text()

    assert 'click=(175,108,448)' in text
    assert 'start=' not in text
    assert 'trajectory=' not in text


def test_summarizer_folds_tool_change_into_next_stroke():
    events = [
        {
            'id': 1,
            'event': 'tool_change',
            'timestamp': '2026-05-13T13:55:14.904',
            'tool': 'erase',
            'segment': 'Segment_1',
        },
        {
            'id': 2,
            'event': 'press',
            'timestamp': '2026-05-13T13:55:16.239',
            'ijk': [210, 384, 448],
            'view': 'Red',
            'tool': 'erase',
            'segment': 'Segment_1',
            'brush_mm': 19.0838,
        },
        {
            'id': 3,
            'event': 'release',
            'timestamp': '2026-05-13T13:55:17.132',
            'ijk': [140, 331, 448],
            'view': 'Red',
        },
    ]

    text = TimeLogSummarizer(_log(events)).export_text()

    assert text.splitlines()[0] == (
        '(13:55:14–13:55:17: Volume=tissuerange.nii.gz, '
        'Segment=Segment_1, View=Red, Tool=erase, brush_mm=19.0838, '
        'slice=448, start=(210,384,448), end=(140,331,448))'
    )


def test_summarizer_groups_volume_navigation_with_initial_volume():
    events = [
        {
            'id': 1,
            'event': 'volume_change',
            'timestamp': '2026-05-13T13:55:19.187',
            'volume': 'PX10013.nii.gz',
        },
        {
            'id': 2,
            'event': 'volume_change',
            'timestamp': '2026-05-13T13:55:19.783',
            'volume': 'tissuerange.nii.gz',
        },
    ]

    text = TimeLogSummarizer(_log(events)).export_text()

    assert text == (
        '(13:55:19: Volume switch/navigation, '
        'tissuerange.nii.gz → PX10013.nii.gz → tissuerange.nii.gz)'
    )


def test_summarizer_groups_slice_navigation():
    events = [
        {'id': 1, 'event': 'slice_change', 'timestamp': '2026-05-13T13:55:43.000', 'view': 'Red', 'slice': 447},
        {'id': 2, 'event': 'slice_change', 'timestamp': '2026-05-13T13:55:44.000', 'view': 'Red', 'slice': 446},
        {'id': 3, 'event': 'slice_change', 'timestamp': '2026-05-13T13:55:47.000', 'view': 'Red', 'slice': 445},
    ]

    text = TimeLogSummarizer(_log(events)).export_text()

    assert text == '(13:55:43–13:55:47: View switch/navigation, Red slice 447 → 446 → 445)'


def test_summarizer_groups_point_drag_moves():
    events = [
        {
            'id': 1,
            'event': 'point_drag_move',
            'timestamp': '2026-05-13T13:55:40.000',
            'view': 'Red',
            'segment': 'Segment_3',
            'point': '1',
            'point_name': 'Segment_3-neg-1',
            'ijk': [10, 20, 445],
        },
        {
            'id': 2,
            'event': 'point_drag_move',
            'timestamp': '2026-05-13T13:55:41.000',
            'view': 'Red',
            'segment': 'Segment_3',
            'point': '1',
            'point_name': 'Segment_3-neg-1',
            'ijk': [30, 40, 445],
        },
    ]

    result = TimeLogSummarizer(_log(events)).export()
    text = result['text'][0]

    assert text.splitlines()[0] == (
        '(13:55:40–13:55:41: Volume=tissuerange.nii.gz, '
        'Segment=Segment_3, View=Red, Point drag=Segment_3-neg-1, '
        'start=(10,20,445), end=(30,40,445))'
    )
    assert text.splitlines()[1] == (
        '  trajectory=[(10,20,445), (30,40,445)]'
    )
    assert result['spans'][0]['trajectory'] == [
        [10, 20, 445], [30, 40, 445],
    ]


def test_summarizer_supports_point_add_delete_and_final_move():
    events = [
        {
            'id': 1,
            'event': 'place',
            'timestamp': '2026-05-13T13:55:20.000',
            'view': 'Red',
            'segment': 'Segment_4',
            'point': 'p1',
            'point_name': 'Segment_4-pos-1',
            'negative': False,
            'ijk': [11, 22, 333],
        },
        {
            'id': 2,
            'event': 'replace',
            'timestamp': '2026-05-13T13:55:21.000',
            'view': 'Red',
            'segment': 'Segment_4',
            'point': 'p1',
            'point_name': 'Segment_4-pos-1',
            'negative': False,
            'ijk': [12, 23, 333],
        },
        {
            'id': 3,
            'event': 'remove',
            'timestamp': '2026-05-13T13:55:22.000',
            'view': 'Red',
            'segment': 'Segment_4',
            'point': 'p1',
            'point_name': 'Segment_4-pos-1',
            'negative': False,
            'ijk': [12, 23, 333],
        },
    ]

    result = TimeLogSummarizer(_log(events)).export()
    text = result['text']

    assert result['spans'][0]['type'] == 'place'
    assert 'Point place=Segment_4-pos-1' in text[0]
    assert 'ijk=(11,22,333)' in text[0]
    assert result['spans'][1]['type'] == 'replace'
    assert 'Point replace=Segment_4-pos-1' in text[1]
    assert 'ijk=(12,23,333)' in text[1]
    assert result['spans'][2]['type'] == 'remove'
    assert 'Point remove=Segment_4-pos-1' in text[2]
    assert 'ijk=(12,23,333)' in text[2]


def test_summarizer_collapses_point_click_and_place():
    events = [
        {
            'id': 1,
            'event': 'press',
            'timestamp': '2026-05-13T14:24:22.100',
            'ijk': [478, 258, 448],
            'view': 'Red',
            'tool': 'point',
            'segment': 'Segment_2',
            'brush_mm': 9.5678,
        },
        {
            'id': 2,
            'event': 'release',
            'timestamp': '2026-05-13T14:24:22.200',
            'ijk': [478, 258, 448],
            'view': 'Red',
        },
        {
            'id': 3,
            'event': 'place',
            'timestamp': '2026-05-13T14:24:22.300',
            'view': 'Red',
            'segment': 'Segment_2',
            'point': '2',
            'point_name': 'Segment_2-pos-2',
            'negative': False,
            'ijk': [478, 258, 448],
        },
    ]

    result = TimeLogSummarizer(_log(events)).export()

    assert len(result['spans']) == 1
    assert result['spans'][0]['type'] == 'place'
    assert result['spans'][0]['click'] == [478, 258, 448]
    assert 'Tool=point' not in result['text'][0]
    assert 'Point place=Segment_2-pos-2' in result['text'][0]
    assert 'negative=False' in result['text'][0]
    assert 'ijk=(478,258,448)' in result['text'][0]
    assert 'click=(478,258,448)' in result['text'][0]
