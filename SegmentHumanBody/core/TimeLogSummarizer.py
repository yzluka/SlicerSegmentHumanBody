"""Second-stage summarizer for compact annotation-process logs.

Input is the semantic ``SegmentHumanBody.annotation_process`` JSON produced by
``TimeLogInterpreter``. Output is a denser, human-readable activity summary that
groups strokes and navigation runs while carrying forward volume/tool/segment
state.
"""

from __future__ import annotations

import datetime


INPUT_TYPE = 'SegmentHumanBody.annotation_process'
EXPORT_TYPE = 'SegmentHumanBody.annotation_summary'

DASH = '\u2013'
ARROW = '\u2192'
MISSING = object()


class TimeLogSummarizer:
    """Summarize semantic recording events into higher-level activity spans."""

    def __init__(self, semantic_log: dict):
        if not isinstance(semantic_log, dict):
            raise TypeError('semantic_log must be a dict')
        if semantic_log.get('type') != INPUT_TYPE:
            raise ValueError(
                f'Expected type {INPUT_TYPE!r}, got {semantic_log.get("type")!r}')
        self._log = semantic_log
        self._events = [
            e for e in (semantic_log.get('events') or [])
            if isinstance(e, dict)
        ]
        meta = semantic_log.get('metadata') or {}
        self._state = {
            'volume': (meta.get('initial_volume') or (meta.get('volume') or {}).get('name')),
            'tool': None,
            'segment': None,
            'view': None,
            'slice': None,
            'brush_mm': None,
        }

    def export(self) -> dict:
        spans = self.spans()
        return {
            'type': EXPORT_TYPE,
            'metadata': dict(self._log.get('metadata') or {}),
            'spans': spans,
            'text': [_span_text(span) for span in spans],
        }

    def export_text(self) -> str:
        export = self.export()
        spans = export.get('spans', [])
        texts = export.get('text', [])

        meta = self._log.get('metadata') or {}
        start_time = meta.get('start_time')
        header = ''
        if start_time:
            try:
                dt = datetime.datetime.fromisoformat(start_time)
                header = f'Recording: {dt.strftime("%Y-%m-%d %H:%M:%S")}\n'
                current_date = dt.strftime('%Y-%m-%d')
            except ValueError:
                header = f'Recording: {start_time}\n'
                current_date = str(start_time)[:10]
        else:
            current_date = None

        parts = []
        for span, text in zip(spans, texts):
            span_date = _date_of(span.get('start_time'))
            if span_date and span_date != current_date:
                if current_date is not None:
                    parts.append(f'--- {span_date} ---')
                current_date = span_date
            parts.append(text)

        body = '\n\n'.join(parts)
        return (header + '\n' + body) if header else body

    def spans(self) -> list[dict]:
        spans = []
        i = 0
        while i < len(self._events):
            ev = self._events[i]
            kind = ev.get('event')
            if kind == 'volume_change':
                span, i = self._consume_volume_navigation(i)
            elif kind == 'slice_change':
                span, i = self._consume_slice_navigation(i)
            elif kind == 'tool_change':
                if (
                        ev.get('tool') is not None and
                        i + 1 < len(self._events) and
                        self._events[i + 1].get('event') == 'press'):
                    self._state['tool'] = ev.get('tool')
                    if ev.get('segment') is not None:
                        self._state['segment'] = ev.get('segment')
                    if self._is_point_click_before_place(i + 1):
                        span, i = self._consume_point_click_place(i + 1)
                    elif self._is_point_drag_start(i + 1):
                        span, i = self._consume_full_point_drag(i + 1)
                    else:
                        span, i = self._consume_stroke(i + 1, start_event=ev)
                else:
                    span, i = self._consume_tool_change(i)
            elif kind == 'press':
                if self._is_point_click_before_place(i):
                    span, i = self._consume_point_click_place(i)
                elif self._is_point_drag_start(i):
                    span, i = self._consume_full_point_drag(i)
                else:
                    span, i = self._consume_stroke(i)
            elif kind == 'point_drag_move':
                span, i = self._consume_point_drag(i)
            elif kind in ('place', 'replace', 'remove', 'point_drag_start'):
                span, i = self._consume_point_event(i)
            elif kind in ('spx_brush_fill', 'spx_erase_fill'):
                span, i = self._consume_spx_stroke(i)
            elif kind == 'fill_hole':
                span, i = self._consume_fill_hole(i)
            elif kind in ('spx_boundary_on', 'spx_boundary_off'):
                span, i = self._consume_spx_boundary(i)
            elif kind in ('model_family_changed', 'model_variant_changed',
                          'model_confirmed'):
                span, i = self._consume_model_change(i)
            else:
                i += 1
                continue
            if span is not None:
                spans.append(span)
        return spans

    def _consume_volume_navigation(self, i):
        start = self._events[i]
        old_volume = self._state.get('volume')
        chain = [old_volume] if old_volume else []
        j = i
        while j < len(self._events) and self._events[j].get('event') == 'volume_change':
            volume = self._events[j].get('volume')
            if volume is not None:
                chain.append(volume)
                self._state['volume'] = volume
            j += 1
        end = self._events[j - 1]
        text = (
            f'({_time_range(start, end)}: Volume switch/navigation, '
            f'{_join_chain(chain)})'
        )
        return _span('volume_navigation', start, end, text, {'volumes': chain}), j

    def _consume_slice_navigation(self, i):
        start = self._events[i]
        view = start.get('view') or self._state.get('view')
        slices = []
        j = i
        while j < len(self._events):
            ev = self._events[j]
            if ev.get('event') != 'slice_change':
                break
            if (ev.get('view') or view) != view:
                break
            if ev.get('slice') is not None:
                slices.append(int(ev['slice']))
                self._state['slice'] = int(ev['slice'])
            if ev.get('view'):
                self._state['view'] = ev.get('view')
            j += 1
        end = self._events[j - 1]
        text = (
            f'({_time_range(start, end)}: View switch/navigation, '
            f'{view} slice {_join_chain(slices)})'
        )
        return _span('slice_navigation', start, end, text, {
            'view': view,
            'slices': slices,
        }), j

    def _consume_tool_change(self, i):
        ev = self._events[i]
        self._state['tool'] = ev.get('tool')
        if ev.get('segment') is not None:
            self._state['segment'] = ev.get('segment')
        details = self._context_parts(
            segment=self._state.get('segment'),
            view=self._state.get('view'),
            tool=self._state.get('tool'),
        )
        text = f'({_time_point(ev)}: {", ".join(details)})'
        return _span('tool_change', ev, ev, text, {
            'volume': self._state.get('volume'),
            'segment': self._state.get('segment'),
            'view': self._state.get('view'),
            'tool': self._state.get('tool'),
        }), i + 1

    def _consume_stroke(self, i, start_event=None):
        press = self._events[i]
        if press.get('tool') is not None:
            self._state['tool'] = press.get('tool')
        if press.get('segment') is not None:
            self._state['segment'] = press.get('segment')
        if press.get('view') is not None:
            self._state['view'] = press.get('view')
        if press.get('brush_mm') is not None:
            self._state['brush_mm'] = press.get('brush_mm')
        if _slice_from_ijk(press.get('ijk')) is not None:
            self._state['slice'] = _slice_from_ijk(press.get('ijk'))

        j = i + 1
        hold = None
        release = None
        if j < len(self._events) and self._events[j].get('event') == 'hold':
            hold = self._events[j]
            j += 1
        if j < len(self._events) and self._events[j].get('event') == 'release':
            release = self._events[j]
            j += 1

        end_event = release or hold or press
        start_ijk = press.get('ijk')
        trajectory = []
        if start_ijk is not None:
            trajectory.append(start_ijk)
        if hold is not None:
            trajectory.extend(_trajectory_ijk(hold))
        end_ijk = (
            release.get('ijk') if release is not None else
            _last_ijk(hold) if hold is not None else
            start_ijk
        )
        if end_ijk is not None and (not trajectory or not _same_ijk(trajectory[-1], end_ijk)):
            trajectory.append(end_ijk)
        slice_idx = _slice_from_ijk(start_ijk) or _slice_from_ijk(end_ijk) or self._state.get('slice')
        if slice_idx is not None:
            self._state['slice'] = slice_idx

        tool = self._state.get('tool')
        brush_mm = self._state.get('brush_mm') if tool in ('brush', 'erase') else None
        details = self._context_parts(
            segment=self._state.get('segment'),
            view=self._state.get('view'),
            tool=tool,
            brush_mm=brush_mm,
            slice_idx=slice_idx,
        )
        is_click = hold is None and _same_ijk(start_ijk, end_ijk)
        if is_click:
            details.append(f'click={_fmt_ijk(start_ijk)}')
            span_type = 'click'
        else:
            details.append(f'start={_fmt_ijk(start_ijk)}')
            details.append(f'end={_fmt_ijk(end_ijk)}')
            span_type = 'stroke'
        start_for_span = start_event or press
        text = f'({_time_range(start_for_span, end_event)}: {", ".join(details)})'
        return _span(span_type, start_for_span, end_event, text, {
            'volume': self._state.get('volume'),
            'segment': self._state.get('segment'),
            'view': self._state.get('view'),
            'tool': tool,
            'brush_mm': brush_mm,
            'slice': slice_idx,
            'start': start_ijk,
            'end': end_ijk,
            'click': start_ijk if is_click else None,
            'trajectory': trajectory if not is_click and len(trajectory) > 1 else None,
        }), j

    def _is_point_click_before_place(self, i):
        if i + 2 >= len(self._events):
            return False
        press = self._events[i]
        release = self._events[i + 1]
        place = self._events[i + 2]
        if press.get('event') != 'press':
            return False
        if release.get('event') != 'release' or place.get('event') != 'place':
            return False
        if press.get('tool') != 'point':
            return False
        if not _same_ijk(press.get('ijk'), release.get('ijk')):
            return False
        if not _same_ijk(press.get('ijk'), place.get('ijk')):
            return False
        if place.get('segment') and press.get('segment') and place.get('segment') != press.get('segment'):
            return False
        return True

    def _consume_point_click_place(self, i):
        press = self._events[i]
        place = self._events[i + 2]
        if press.get('tool') is not None:
            self._state['tool'] = press.get('tool')
        if place.get('segment') is not None:
            self._state['segment'] = place.get('segment')
        elif press.get('segment') is not None:
            self._state['segment'] = press.get('segment')
        if place.get('view') is not None:
            self._state['view'] = place.get('view')
        elif press.get('view') is not None:
            self._state['view'] = press.get('view')
        if _slice_from_ijk(place.get('ijk')) is not None:
            self._state['slice'] = _slice_from_ijk(place.get('ijk'))
        return self._point_event_span(
            place,
            start_event=press,
            extra={'click': press.get('ijk')},
        ), i + 3

    def _is_point_drag_start(self, i):
        if i + 1 >= len(self._events):
            return False
        return (self._events[i].get('event') == 'press' and
                self._events[i + 1].get('event') == 'point_drag_move')

    def _consume_full_point_drag(self, i):
        press = self._events[i]
        if press.get('segment') is not None:
            self._state['segment'] = press.get('segment')
        if press.get('view') is not None:
            self._state['view'] = press.get('view')

        j = i + 1
        drag_points = []
        replace_ev = None
        point_name = None
        point_id = None

        while j < len(self._events):
            ev = self._events[j]
            ek = ev.get('event')
            if ek == 'point_drag_move':
                if ev.get('ijk') is not None:
                    drag_points.append(ev['ijk'])
                if point_name is None:
                    point_name = ev.get('point_name')
                    point_id = ev.get('point')
                    if ev.get('segment') is not None:
                        self._state['segment'] = ev.get('segment')
                    if ev.get('view') is not None:
                        self._state['view'] = ev.get('view')
                j += 1
            elif ek in ('hold', 'release'):
                j += 1
            elif ek == 'replace':
                replace_ev = ev
                j += 1
                break
            else:
                break

        end_ev = replace_ev or self._events[j - 1]
        if replace_ev is not None:
            if replace_ev.get('segment') is not None:
                self._state['segment'] = replace_ev.get('segment')
            if replace_ev.get('view') is not None:
                self._state['view'] = replace_ev.get('view')

        segment = self._state.get('segment')
        view = self._state.get('view')
        details = self._context_parts(segment=segment, view=view)
        details.append(f'Point drag={point_name or point_id}')
        end_ijk = replace_ev.get('ijk') if replace_ev else (drag_points[-1] if drag_points else None)
        if drag_points:
            details.append(f'start={_fmt_ijk(drag_points[0])}')
            details.append(f'end={_fmt_ijk(end_ijk)}')
        text = f'({_time_range(press, end_ev)}: {", ".join(details)})'
        return _span('point_drag', press, end_ev, text, {
            'volume': self._state.get('volume'),
            'segment': segment,
            'view': view,
            'point': point_id,
            'point_name': point_name,
            'start': drag_points[0] if drag_points else None,
            'end': end_ijk,
            'trajectory': drag_points if len(drag_points) > 1 else None,
        }), j

    def _consume_point_drag(self, i):
        start = self._events[i]
        key = (start.get('segment'), start.get('point'), start.get('point_name'))
        points = []
        j = i
        while j < len(self._events):
            ev = self._events[j]
            ek = ev.get('event')
            if ek == 'hold':
                j += 1
                continue
            if ek != 'point_drag_move':
                break
            ev_key = (ev.get('segment'), ev.get('point'), ev.get('point_name'))
            if ev_key != key:
                break
            if ev.get('ijk') is not None:
                points.append(ev.get('ijk'))
            j += 1
        end = self._events[j - 1]
        if start.get('view') is not None:
            self._state['view'] = start.get('view')
        if start.get('segment') is not None:
            self._state['segment'] = start.get('segment')
        details = self._context_parts(
            segment=start.get('segment'),
            view=start.get('view') or self._state.get('view'),
        )
        details.append(f'Point drag={start.get("point_name") or start.get("point")}')
        if points:
            details.append(f'start={_fmt_ijk(points[0])}')
            details.append(f'end={_fmt_ijk(points[-1])}')
        text = f'({_time_range(start, end)}: {", ".join(details)})'
        return _span('point_drag', start, end, text, {
            'segment': start.get('segment'),
            'point': start.get('point'),
            'point_name': start.get('point_name'),
            'start': points[0] if points else None,
            'end': points[-1] if points else None,
            'trajectory': points if len(points) > 1 else None,
        }), j

    def _consume_point_event(self, i):
        ev = self._events[i]
        span = self._point_event_span(ev)
        return span, i + 1

    def _point_event_span(self, ev, start_event=None, extra=None):
        if ev.get('view') is not None:
            self._state['view'] = ev.get('view')
        if ev.get('segment') is not None:
            self._state['segment'] = ev.get('segment')
        label = {
            'place': 'Point place',
            'replace': 'Point replace',
            'remove': 'Point remove',
            'point_drag_start': 'Point drag start',
        }.get(ev.get('event'), ev.get('event'))
        details = self._context_parts(
            segment=ev.get('segment'),
            view=ev.get('view') or self._state.get('view'),
        )
        details.append(f'{label}={ev.get("point_name") or ev.get("point")}')
        if ev.get('negative') is not None:
            details.append(f'negative={bool(ev.get("negative"))}')
        if ev.get('ijk') is not None:
            details.append(f'ijk={_fmt_ijk(ev.get("ijk"))}')
        merged = dict(extra or {})
        if merged.get('click') is not None:
            details.append(f'click={_fmt_ijk(merged["click"])}')
        start = start_event or ev
        text = f'({_time_range(start, ev)}: {", ".join(details)})'
        data = {
            'segment': ev.get('segment'),
            'point': ev.get('point'),
            'point_name': ev.get('point_name'),
            'negative': ev.get('negative'),
            'ijk': ev.get('ijk'),
        }
        data.update(merged)
        return _span(ev.get('event'), start, ev, text, data)

    def _consume_spx_stroke(self, i):
        first = self._events[i]
        kind = first.get('event')
        merge_key = (kind, first.get('segment_id'), first.get('slice_idx'), first.get('view'))
        labels = []
        total_delta = 0
        j = i
        while j < len(self._events):
            ev = self._events[j]
            this_key = (ev.get('event'), ev.get('segment_id'), ev.get('slice_idx'), ev.get('view'))
            if this_key != merge_key:
                break
            label_id = ev.get('label_id')
            if label_id is not None:
                labels.append(label_id)
            delta = ev.get('delta_pixels')
            if delta is not None:
                total_delta += delta
            j += 1
        end = self._events[j - 1]
        span_type = 'spx_brush_stroke' if kind == 'spx_brush_fill' else 'spx_erase_stroke'
        ctx = self._context_parts(segment=first.get('segment_id'), view=first.get('view'),
                                  slice_idx=first.get('slice_idx'))
        text = f'({_time_range(first, end)}: {span_type} labels={labels} Δ={total_delta}px {", ".join(ctx)})'
        return _span(span_type, first, end, text, {
            'labels': labels, 'total_delta_pixels': total_delta,
            'slice_idx': first.get('slice_idx'), 'view': first.get('view'),
            'segment_id': first.get('segment_id'),
            'model_key': first.get('model_key'),
        }), j

    def _consume_fill_hole(self, i):
        ev = self._events[i]
        ctx = self._context_parts(segment=ev.get('segment_id'), view=ev.get('view'),
                                  slice_idx=ev.get('slice_idx'))
        text = f'({_time_point(ev)}: fill_hole Δ={ev.get("delta_pixels")}px {", ".join(ctx)})'
        return _span('fill_hole', ev, ev, text, {
            'slice_idx': ev.get('slice_idx'), 'view': ev.get('view'),
            'delta_pixels': ev.get('delta_pixels'),
            'segment_id': ev.get('segment_id'),
        }), i + 1

    def _consume_spx_boundary(self, i):
        first = self._events[i]
        kind = first.get('event')
        j = i + 1
        end = first
        if kind == 'spx_boundary_on' and j < len(self._events):
            if self._events[j].get('event') == 'spx_boundary_off':
                end = self._events[j]
                j += 1
        text = f'({_time_range(first, end)}: spx_boundary_inspection view={first.get("view")})'
        return _span('spx_boundary_inspection', first, end, text, {
            'view': first.get('view'), 'model_key': first.get('model_key'),
        }), j

    def _consume_model_change(self, i):
        ev = self._events[i]
        kind = ev.get('event')
        text = f'({_time_point(ev)}: {kind} family={ev.get("family")} variant={ev.get("variant")})'
        return _span('model_change', ev, ev, text, {
            'family': ev.get('family'), 'variant': ev.get('variant'),
        }), i + 1

    def _context_parts(self, *, segment=None, view=None, tool=MISSING,
                       brush_mm=None, slice_idx=None):
        parts = [f'Volume={_fmt_value(self._state.get("volume"))}']
        if segment is not None:
            parts.append(f'Segment={segment}')
        if view is not None:
            parts.append(f'View={view}')
        if tool is not MISSING:
            parts.append(f'Tool={_fmt_value(tool)}')
        if brush_mm is not None:
            parts.append(f'brush_mm={brush_mm}')
        if slice_idx is not None:
            parts.append(f'slice={int(slice_idx)}')
        return parts


def _span(kind, start, end, text, data):
    span = {
        'type': kind,
        'start_time': _first_timestamp(start),
        'end_time': _last_timestamp(end),
        'start_id': start.get('id'),
        'end_id': end.get('id'),
        'text': text,
    }
    span.update({k: v for k, v in data.items() if v is not None})
    return span


def _span_text(span):
    text = span['text']
    trajectory = span.get('trajectory')
    if trajectory:
        text += '\n  trajectory=' + _fmt_trajectory(trajectory)
    return text


def _time_range(start, end):
    a = _time_of_day(_first_timestamp(start))
    b = _time_of_day(_last_timestamp(end))
    return a if a == b else f'{a}{DASH}{b}'


def _time_point(event):
    return _time_of_day(_first_timestamp(event))


def _first_timestamp(event):
    ts = event.get('timestamp')
    if isinstance(ts, list):
        return ts[0] if ts else None
    return ts


def _last_timestamp(event):
    ts = event.get('timestamp')
    if isinstance(ts, list):
        return ts[-1] if ts else None
    return ts


def _time_of_day(timestamp):
    if not timestamp:
        return '??:??:??'
    try:
        return datetime.datetime.fromisoformat(timestamp).strftime('%H:%M:%S')
    except ValueError:
        return str(timestamp)[11:19] if len(str(timestamp)) >= 19 else str(timestamp)


def _fmt_value(value):
    return 'null' if value is None else str(value)


def _fmt_ijk(ijk):
    if ijk is None:
        return '(?, ?, ?)'
    return '(' + ','.join(str(int(round(float(v)))) for v in ijk[:3]) + ')'


def _fmt_trajectory(points):
    return '[' + ', '.join(_fmt_ijk(point) for point in points) + ']'


def _same_ijk(a, b):
    return a is not None and b is not None and list(a[:3]) == list(b[:3])


def _last_ijk(event):
    if not event:
        return None
    ijk = event.get('ijk')
    if isinstance(ijk, list) and ijk and isinstance(ijk[0], list):
        return ijk[-1]
    return ijk


def _trajectory_ijk(event):
    ijk = (event or {}).get('ijk')
    if isinstance(ijk, list) and ijk and isinstance(ijk[0], list):
        return list(ijk)
    return []


def _slice_from_ijk(ijk):
    if isinstance(ijk, list) and len(ijk) >= 3 and not isinstance(ijk[0], list):
        return int(round(float(ijk[2])))
    return None


def _join_chain(values):
    return f' {ARROW} '.join(str(v) for v in values if v is not None)


def _date_of(timestamp):
    if not timestamp:
        return None
    try:
        return datetime.datetime.fromisoformat(str(timestamp)).strftime('%Y-%m-%d')
    except ValueError:
        s = str(timestamp)
        return s[:10] if len(s) >= 10 else None
