"""Offline interpreter: SegmentHumanBody raw_input -> annotation_process log.

Standalone — no dependency on _mouse_recorder or Slicer.

Raw log format (SegmentHumanBody.raw_input):
  metadata.initial_visual_state[view].xy_to_ijk  — 4x4 flat row-major matrix
  metadata.volume.dimensions
  events[]:
    mouse boundary  — {event:"mouse", timestamp, mouse_state:"press"|"release", xy, view, slice, tool, segment, diameter_mm}
    mouse trajectory — {event:"mouse", timestamps:[], xy:[], mouse_state:"hold"|"idle", view, slice, ...}
    point_placed    — {event:"point_placed", timestamp, view, ijk, segment, point, ...}
    point_replaced  — {event:"point_replaced", ...}
    point_removed   — {event:"point_removed",  ...}
    point_drag_start — {event:"point_drag_start", ijk, point, ...} origin captured for replace
    point_drag_move — trajectory; skipped in compact output
    tool_selected   — {event:"tool_selected", tool, segment?} → tool_change (unless after segment_created)
    segment_removed — {event:"segment_removed", segment, seg_name?} → segment_removed
    brush_parameters_changed — updates brush state

Compact output (SegmentHumanBody.annotation_process):
  Boundary events:
    press:        {id, event:"press",   timestamp, ijk, view, tool, segment, brush_mm}
    hold:         {id, event:"hold",    timestamp:[...], ijk:[[...]], view}
    release:      {id, event:"release", timestamp, ijk, view}
    place:        {id, event:"place",   timestamp, ijk, view, segment, point, point_name, point_index, negative}
    replace:      {id, event:"replace", ...same as place...}
    remove:       {id, event:"remove",  ...same as place...}
  Delta / lifecycle events:
    slice_change:    {id, event:"slice_change",    timestamp, view, slice}
    tool_change:     {id, event:"tool_change",     timestamp, tool, segment?}
                     Sourced from explicit tool_selected raw events. Suppressed when
                     the tool_selected was auto-triggered by a new segment.
    brush_change:    {id, event:"brush_change",    timestamp, view, tool, brush_mm?, sphere?}
                     brush_mm and sphere present only when their respective value changed.
    segment_removed: {id, event:"segment_removed", timestamp, segment, seg_name?}

  segment, tool, and brush_mm are written only on press (delta pattern).
  Idle/hover trajectory is omitted entirely.
"""

RAW_TYPE = 'SegmentHumanBody.raw_input'
EXPORT_TYPE = 'SegmentHumanBody.annotation_process'

_ANNOTATION_TOOLS = frozenset(('brush', 'erase'))
_POINT_BOUNDARY_EVENTS = frozenset(('point_placed', 'point_replaced', 'point_removed'))
_POINT_EVENT_MAP = {
    'point_placed': 'place',
    'point_replaced': 'replace',
    'point_removed': 'remove',
}


class TimeLogInterpreter:
    """Convert a ``*_raw.json`` log to the compact annotation-process log."""

    def __init__(self, raw_log: dict):
        if not isinstance(raw_log, dict):
            raise TypeError('raw_log must be a dict')
        if raw_log.get('type') != RAW_TYPE:
            raise ValueError(
                f'Expected type {RAW_TYPE!r}, got {raw_log.get("type")!r}')
        meta = raw_log.get('metadata') or {}
        self._raw = raw_log
        self._meta = meta
        self._view_mats = _load_view_matrices(meta)
        self._dims = _load_dims(meta)
        self._brush_mm = _load_brush_mm(meta)

    def export(self) -> dict:
        raws = [e for e in (self._raw.get('events') or []) if isinstance(e, dict)]
        companions = _find_point_companions(raws)
        events = []
        export_id = 1
        last_slice = {}       # view -> int
        point_positions = {}  # ('id', pt_id) | ('name', pt_name) -> last known ijk

        for i, raw in enumerate(raws):
            if i in companions:
                continue
            evt = raw.get('event')
            if not evt:
                continue

            view = raw.get('view')
            s = raw.get('slice')
            if s is not None and view:
                new_slice = int(s)
                if view in last_slice and last_slice[view] != new_slice:
                    ts = raw.get('timestamp') or (raw.get('timestamps') or [''])[0]
                    events.append({
                        'id': export_id,
                        'event': 'slice_change',
                        'timestamp': ts,
                        'view': view,
                        'slice': new_slice,
                    })
                    export_id += 1
                last_slice[view] = new_slice

            if evt == 'tool_selected':
                if not _is_segment_creation_tool_select(raws, i):
                    tool = raw.get('tool')
                    ev = {'id': export_id, 'event': 'tool_change',
                          'timestamp': raw.get('timestamp'), 'tool': tool}
                    segment = raw.get('segment')
                    if segment:
                        ev['segment'] = segment
                    export_id += 1
                    events.append(ev)

            elif evt == 'mouse':
                for ev in self._from_mouse(raw, last_slice):
                    ev['id'] = export_id
                    export_id += 1
                    events.append(ev)

            elif evt == 'brush_parameters_changed':
                _apply_brush_update(self._brush_mm, raw)
                ev = _brush_change_event(raw)
                if ev is not None:
                    ev['id'] = export_id
                    export_id += 1
                    events.append(ev)

            elif evt in _POINT_BOUNDARY_EVENTS:
                pt_id = raw.get('point')
                pt_name = raw.get('point_name')
                ijk_start = None
                if evt == 'point_replaced':
                    ijk_start = (point_positions.get(('id', pt_id))
                                 or point_positions.get(('name', pt_name)))
                ev = self._from_point_event(raw, last_slice, ijk_start=ijk_start)
                if ev is not None:
                    ev['id'] = export_id
                    export_id += 1
                    events.append(ev)
                    if ev.get('ijk') and evt in ('point_placed', 'point_replaced'):
                        if pt_id:
                            point_positions[('id', pt_id)] = ev['ijk']
                        if pt_name:
                            point_positions[('name', pt_name)] = ev['ijk']
                    elif evt == 'point_removed':
                        point_positions.pop(('id', pt_id), None)
                        point_positions.pop(('name', pt_name), None)

            elif evt == 'segment_removed':
                ev = {'id': export_id, 'event': 'segment_removed',
                      'timestamp': raw.get('timestamp')}
                segment = raw.get('segment')
                if segment:
                    ev['segment'] = segment
                seg_name = raw.get('seg_name')
                if seg_name:
                    ev['seg_name'] = seg_name
                export_id += 1
                events.append(ev)
            # point_drag_move, point_drag_start, segment_created/selected/renamed,
            # volume_changed, etc.: not in annotation-process log

        return {
            'type': EXPORT_TYPE,
            'metadata': _compact_metadata(self._meta),
            'events': events,
        }

    # ── Per-event converters ──────────────────────────────────────────────────

    def _from_mouse(self, raw, last_slice=None):
        view = raw.get('view')
        state = raw.get('mouse_state')
        slice_idx = raw.get('slice')
        if slice_idx is None and last_slice and view:
            slice_idx = last_slice.get(view)

        if isinstance(raw.get('timestamps'), list):
            # Trajectory block: keep only held-button annotation strokes.
            if state != 'hold':
                return
            ijk_list = []
            ts_list = []
            for ts_str, xy in zip(raw['timestamps'], raw.get('xy') or []):
                ijk = self._to_ijk(view, xy, slice_idx)
                if ijk is not None:
                    ijk_list.append(ijk)
                    ts_list.append(ts_str)
            if not ijk_list:
                return
            ev = {'event': 'hold', 'timestamp': ts_list, 'ijk': ijk_list}
            if view:
                ev['view'] = view
            yield ev
        else:
            # Single boundary event.
            ts_str = raw.get('timestamp')
            xy = raw.get('xy')
            if ts_str is None or xy is None:
                return
            ijk = self._to_ijk(view, xy, slice_idx)
            if ijk is None:
                return

            if state == 'press':
                tool = raw.get('tool')
                segment = raw.get('segment')
                diameter_mm = raw.get('diameter_mm')
                brush_mm = (
                    float(diameter_mm) / 2.0 if diameter_mm is not None
                    else self._brush_mm.get(view) if view else None
                )
                ev = {'event': 'press', 'timestamp': ts_str, 'ijk': ijk}
                if view:
                    ev['view'] = view
                if tool:
                    ev['tool'] = tool
                if segment:
                    ev['segment'] = segment
                if brush_mm is not None and tool in _ANNOTATION_TOOLS:
                    ev['brush_mm'] = brush_mm
                yield ev
            elif state == 'release':
                ev = {'event': 'release', 'timestamp': ts_str, 'ijk': ijk}
                if view:
                    ev['view'] = view
                yield ev
            # Other states (idle single boundary): omitted.

    def _point_ijk(self, raw, last_slice=None):
        """Derive IJK for a point event: prefer raw.ijk, fall back to mouse_boundary.xy."""
        ijk = raw.get('ijk')
        if ijk is not None:
            return ijk
        view = raw.get('view')
        mb = raw.get('mouse_boundary') or {}
        xy = mb.get('xy')
        if xy is None:
            return None
        slice_idx = raw.get('slice')
        if slice_idx is None and last_slice and view:
            slice_idx = last_slice.get(view)
        return self._to_ijk(view, xy, slice_idx)

    def _from_point_event(self, raw, last_slice=None, ijk_start=None):
        ijk = self._point_ijk(raw, last_slice)
        view = raw.get('view')
        if ijk is None:
            return None

        event_type = _POINT_EVENT_MAP.get(raw.get('event'), raw.get('event'))
        ev = {'event': event_type, 'timestamp': raw.get('timestamp'), 'ijk': ijk}
        if view:
            ev['view'] = view
        for key in ('segment', 'point', 'point_name', 'point_index', 'negative'):
            val = raw.get(key)
            if val is not None:
                ev[key] = val
        if ijk_start is not None:
            ev['ijk_start'] = ijk_start
        return ev

    def _to_ijk(self, view, xy, slice_idx=None):
        if view is None or xy is None:
            return None
        mat = self._view_mats.get(view)
        if mat is None:
            return None
        return _apply_mat(mat, xy, self._dims, slice_idx)


# ── Companion detection ───────────────────────────────────────────────────────

def _find_point_companions(raws):
    """Return indices of mouse press/release events that bracket point events.

    These are suppressed from the compact log: the point semantic event carries
    position and intent; the flanking mouse boundaries are redundant.
    """
    companions = set()
    for i, raw in enumerate(raws):
        evt = raw.get('event')

        if evt == 'point_placed':
            j = i - 1
            for expected in ('release', 'press'):
                while j >= 0:
                    r = raws[j]
                    if (r.get('event') == 'mouse'
                            and r.get('mouse_state') == expected):
                        companions.add(j)
                        j -= 1
                        break
                    j -= 1

        elif evt == 'point_replaced':
            j = i - 1
            found_release = False
            while j >= 0:
                r = raws[j]
                re = r.get('event')
                if re == 'point_drag_move':
                    j -= 1
                    continue
                if re == 'mouse' and isinstance(r.get('timestamps'), list):
                    companions.add(j)
                    j -= 1
                    continue
                if re == 'mouse':
                    rs = r.get('mouse_state')
                    if rs == 'release' and not found_release:
                        companions.add(j)
                        found_release = True
                    elif rs == 'press' and found_release:
                        companions.add(j)
                        break
                j -= 1

    return companions


def _is_segment_creation_tool_select(raws, i):
    """True if the tool_selected at index i was auto-triggered by a new segment.

    Looks backward past any segment_selected events; if the first earlier
    meaningful event is segment_created, the tool_selected is automatic and
    should not produce a tool_change in the compact log.
    """
    j = i - 1
    while j >= 0:
        prev_evt = raws[j].get('event')
        if prev_evt == 'segment_selected':
            j -= 1
            continue
        return prev_evt == 'segment_created'
    return False


# ── Matrix / coordinate helpers ───────────────────────────────────────────────

def _apply_mat(mat, xy, dims, slice_idx=None):
    """Apply a 4x4 flat row-major xy_to_ijk matrix to device point (x, y, 0, 1)."""
    x, y = float(xy[0]), float(xy[1])
    i = mat[0] * x + mat[1] * y + mat[3]
    j = mat[4] * x + mat[5] * y + mat[7]
    k = float(slice_idx) if slice_idx is not None else mat[8] * x + mat[9] * y + mat[11]
    ijk = [int(round(i)), int(round(j)), int(round(k))]
    if dims is not None:
        if not (0 <= ijk[0] < dims[0]
                and 0 <= ijk[1] < dims[1]
                and 0 <= ijk[2] < dims[2]):
            return None
    return ijk


def _load_view_matrices(meta):
    ras_to_ijk = (meta.get('volume') or {}).get('ras_to_ijk')
    result = {}
    for view, data in (meta.get('initial_visual_state') or {}).items():
        mat = data.get('xy_to_ijk')
        if mat and len(mat) == 16:
            result[view] = mat
            continue
        xy_to_ras = data.get('xy_to_ras')
        if xy_to_ras and len(xy_to_ras) == 16 and ras_to_ijk and len(ras_to_ijk) == 16:
            result[view] = _mat4_mul(ras_to_ijk, xy_to_ras)
    return result


def _mat4_mul(a, b):
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            out[r * 4 + c] = sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4))
    return out


def _load_dims(meta):
    dims = (meta.get('volume') or {}).get('dimensions')
    return list(dims) if dims and len(dims) == 3 else None


def _load_brush_mm(meta):
    result = {}
    for view, ctx in (meta.get('initial_handler_context') or {}).items():
        r = ctx.get('brush_radius_mm')
        if r is not None:
            result[view] = float(r)
    return result


def _apply_brush_update(brush_mm, raw):
    view = raw.get('view')
    if view is None:
        return
    r = raw.get('brush_radius_mm')
    if r is not None:
        brush_mm[view] = float(r)
    elif raw.get('diameter_mm') is not None:
        brush_mm[view] = float(raw['diameter_mm']) / 2.0


def _brush_change_event(raw):
    ts = raw.get('timestamp')
    if ts is None:
        return None
    diameter_mm = raw.get('diameter_mm')
    sphere = raw.get('sphere')
    if diameter_mm is None and sphere is None:
        return None
    ev = {'event': 'brush_change', 'timestamp': ts}
    view = raw.get('view')
    if view:
        ev['view'] = view
    tool = raw.get('tool')
    if tool:
        ev['tool'] = tool
    if diameter_mm is not None:
        ev['brush_mm'] = float(diameter_mm) / 2.0
    if sphere is not None:
        ev['sphere'] = bool(sphere)
    return ev


# ── Metadata ──────────────────────────────────────────────────────────────────

def _compact_metadata(meta):
    out = {}
    if meta.get('start_time'):
        out['start_time'] = meta['start_time']
    vol = meta.get('volume')
    if vol:
        out['volume'] = {
            k: vol[k]
            for k in ('name', 'dimensions', 'spacing', 'ijk_to_ras', 'ras_to_ijk')
            if k in vol
        }
    if meta.get('move_thinning'):
        out['move_thinning'] = meta['move_thinning']
    return out
