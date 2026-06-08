"""TimedAnnotatorModel — per-segment point logger with Slicer mirror nodes.

Stateful: accumulates a timestamped log and one vtkMRMLMarkupsFiducialNode per
segment across the session.  The ModelRegistry caches exactly one instance per
session so that switching away from TimedAnnotatorFamily and back preserves the
accumulated annotations.
"""


class TimedAnnotatorModel:
    """Records timed annotation points and mirrors them as Slicer markup nodes."""

    PARAM_HINT: str = ''
    DOC_URL: str | None = None
    REQUIRES_DISTRIBUTIONS: tuple = ()

    # Distinguishable palette cycled across segments (RGB 0–1).
    _PALETTE: tuple = (
        (1.00, 1.00, 0.00),   # yellow
        (0.00, 1.00, 1.00),   # cyan
        (1.00, 0.00, 1.00),   # magenta
        (1.00, 0.55, 0.00),   # orange
        (0.40, 1.00, 0.40),   # lime
        (0.30, 0.70, 1.00),   # sky blue
        (1.00, 0.50, 0.80),   # pink
        (0.70, 0.30, 1.00),   # violet
    )

    def __init__(self):
        self._log: list = []
        self._nodes: dict = {}        # segment_id → vtkMRMLMarkupsFiducialNode
        self._seg_colors: dict = {}   # segment_id → (r, g, b); assigned once per segment
        self._volume_node = None      # most-recently-seen volume; used for lazy IJK export
        self._segmentation_node = None  # most-recently-seen segmentation; used for lazy name export
        # mirror_cp_id → {'segment_id', 'versions': [{'coord_ras', 'timestamp'}], 'alive'}
        self._point_history: dict = {}

    # ------------------------------------------------------------------
    # Family hooks (called by TimedAnnotatorFamily delegates)
    # ------------------------------------------------------------------

    def on_segment_created(self, segment_id: str, seg_name: str, segmentation_node=None) -> None:
        if segmentation_node is not None:
            self._segmentation_node = segmentation_node
        from datetime import datetime
        self._log.append({
            'type':       'segment',
            'segment_id': segment_id,
            'seg_name':   seg_name,
            'timestamp':  datetime.now().isoformat(),
        })

    def on_point_confirmed(self, segment_id: str, ras, cp_id: str, is_negative: bool = False,
                           volume_node=None, segmentation_node=None) -> None:
        if is_negative:
            return
        if volume_node is not None:
            self._volume_node = volume_node
        if segmentation_node is not None:
            self._segmentation_node = segmentation_node
        from datetime import datetime
        ts = datetime.now().isoformat()
        mirror_cp_id = self._mirror_to_node(segment_id, ras)
        self._log.append({
            'type':         'point',
            'segment_id':   segment_id,
            'coord_ras':    list(ras),
            'timestamp':    ts,
            'cp_id':        cp_id,
            'mirror_cp_id': mirror_cp_id,
        })
        self._point_history[mirror_cp_id] = {
            'segment_id': segment_id,
            'versions':   [{'coord_ras': list(ras), 'timestamp': ts}],
            'alive':      True,
        }

    def on_point_undone(self, cp_id: str) -> None:
        """Remove the log entry whose prompt cp_id matches; undo the mirror node too."""
        from datetime import datetime
        for i in range(len(self._log) - 1, -1, -1):
            entry = self._log[i]
            if entry.get('type') == 'point' and entry.get('cp_id') == cp_id:
                mirror_cp_id = entry.get('mirror_cp_id')
                seg_id       = entry.get('segment_id')
                # Mark deleted in history BEFORE removing from log/node so that
                # _on_mirror_point_removed (fired by RemoveNthControlPoint) sees an
                # already-gone log entry and skips its own log-cleanup pass.
                if mirror_cp_id and mirror_cp_id in self._point_history:
                    self._point_history[mirror_cp_id]['versions'].append(
                        {'coord_ras': None, 'timestamp': datetime.now().isoformat()}
                    )
                    self._point_history[mirror_cp_id]['alive'] = False
                del self._log[i]
                if mirror_cp_id and seg_id and seg_id in self._nodes:
                    node = self._nodes[seg_id]
                    idx = node.GetControlPointIndexByID(mirror_cp_id)
                    if idx >= 0:
                        node.RemoveNthControlPoint(idx)
                return

    def sync_visibility(
        self,
        current_seg_id: str,
        current_visible: bool,
        saved_visible: bool,
    ) -> None:
        """Match mirror node visibility to the current/saved checkbox states."""
        import slicer
        for seg_id, node in self._nodes.items():
            if not slicer.mrmlScene.IsNodePresent(node):
                continue
            visible = current_visible if seg_id == current_seg_id else saved_visible
            node.SetDisplayVisibility(int(visible))

    def export_data(self) -> dict:
        """Return annotation data in nested per-segment format.

        Each point entry contains versioned lists (one element per version):

        Structure::

            {
                "segments": {
                    "<segment_id>": {
                        "seg_name": str,
                        "points": {
                            "<mirror_cp_id>": {
                                "alive":      bool,
                                "coord_ras":  [[x0,y0,z0], [x1,y1,z1], ...],
                                "coord_ijk":  [[i0,j0,k0], ..., null],
                                "timestamp":  [t0, t1, ...],
                            },
                            ...
                        }
                    },
                    ...
                }
            }
        """
        log_seg_names = {
            e['segment_id']: e['seg_name']
            for e in self._log
            if e.get('type') == 'segment'
        }
        segments: dict = {}

        def _ensure_segment(seg_id):
            if seg_id not in segments:
                fallback = log_seg_names.get(seg_id, seg_id)
                segments[seg_id] = {
                    'seg_name': self._get_segment_name(seg_id, fallback),
                    'points':   {},
                }

        def _build_point(ras_list, ts_list, alive):
            return {
                'alive':     alive,
                'coord_ras': ras_list,
                'coord_ijk': [
                    self._compute_ijk_from_ras(r) if r is not None else None
                    for r in ras_list
                ],
                'timestamp': ts_list,
            }

        # alive points (from _log — source of truth for what exists in the scene)
        alive_mids: set = set()
        for e in self._log:
            if e.get('type') != 'point':
                continue
            seg_id = e['segment_id']
            mid    = e['mirror_cp_id']
            alive_mids.add(mid)
            _ensure_segment(seg_id)
            hist = self._point_history.get(mid)
            if hist:
                versions = hist['versions']
                ras_list = [v['coord_ras'] for v in versions]
                ts_list  = [v['timestamp'] for v in versions]
            else:
                ras_list = [e['coord_ras']]
                ts_list  = [e['timestamp']]
            segments[seg_id]['points'][mid] = _build_point(ras_list, ts_list, alive=True)

        # deleted points (in _point_history, not in _log)
        for mid, hist in self._point_history.items():
            if mid in alive_mids or hist.get('alive', True):
                continue
            seg_id   = hist['segment_id']
            versions = hist['versions']
            _ensure_segment(seg_id)
            segments[seg_id]['points'][mid] = _build_point(
                [v['coord_ras'] for v in versions],
                [v['timestamp'] for v in versions],
                alive=False,
            )

        return {'segments': segments}

    def _get_segment_name(self, segment_id: str, fallback: str) -> str:
        """Return the current segment name from the segmentation node, or *fallback*."""
        if self._segmentation_node is None:
            return fallback
        try:
            import slicer
            if not slicer.mrmlScene.IsNodePresent(self._segmentation_node):
                return fallback
            seg = self._segmentation_node.GetSegmentation().GetSegment(segment_id)
            if seg is not None:
                return seg.GetName()
        except Exception:
            pass
        return fallback

    def _compute_ijk_from_ras(self, ras) -> list | None:
        """Compute IJK voxel indices from RAS using the stored volume node, or None."""
        if self._volume_node is None:
            return None
        try:
            import slicer
            if not slicer.mrmlScene.IsNodePresent(self._volume_node):
                return None
            import vtk
            import numpy as np
            vtk_mat = vtk.vtkMatrix4x4()
            self._volume_node.GetRASToIJKMatrix(vtk_mat)
            mat = np.array([[vtk_mat.GetElement(r, c) for c in range(4)] for r in range(4)])
            pt_h = np.array([ras[0], ras[1], ras[2], 1.0], dtype=np.float64)
            ijk_h = mat @ pt_h
            return [int(round(ijk_h[i])) for i in range(3)]
        except Exception:
            return None

    def on_export(self, widget) -> None:
        """Open a save-file dialog and write the log as JSON."""
        import json
        import qt
        path = qt.QFileDialog.getSaveFileName(
            None, 'Export Annotation Log', '', 'JSON files (*.json)'
        )
        if not path:
            return
        if not path.endswith('.json'):
            path += '.json'
        with open(path, 'w') as f:
            json.dump(self.export_data(), f, indent=2)
        import slicer
        slicer.util.infoDisplay(f'Annotation log exported to:\n{path}')

    def load_from_json(self, json_data) -> int:
        """Replay a previously exported annotation log.

        Accepts the current nested format
        ``{"segments": {"<id>": {"seg_name": ..., "points": {...}}}}``
        and the legacy flat-list format
        ``[{"segment_id": ..., "coord_ras": ..., "timestamp": ...}, ...]``.
        Returns the number of alive points imported (mirrors placed in scene).
        """
        if isinstance(json_data, dict):
            return self._load_nested(json_data)
        return self._load_flat(json_data)

    def _load_nested(self, data: dict) -> int:
        count = 0
        for seg_id, seg_data in data.get('segments', {}).items():
            if not seg_id:
                continue
            points_raw = seg_data.get('points', {})
            items = points_raw.items() if isinstance(points_raw, dict) else ((None, p) for p in points_raw)
            for _key, point in items:
                alive       = point.get('alive', True)
                coord_ras_r = point.get('coord_ras', [0.0, 0.0, 0.0])
                ts_r        = point.get('timestamp', '')

                # Detect versioned format: coord_ras is a list of lists
                if (isinstance(coord_ras_r, list) and coord_ras_r
                        and isinstance(coord_ras_r[0], list)):
                    ras_list = coord_ras_r
                    ts_list  = ts_r if isinstance(ts_r, list) else [ts_r]
                    last_ras = next((r for r in reversed(ras_list) if r is not None), None)
                else:
                    last_ras = coord_ras_r
                    ras_list = [coord_ras_r]
                    ts_list  = [ts_r]

                if alive and last_ras is not None:
                    last_ts      = ts_list[-1] if ts_list else ''
                    mirror_cp_id = self._mirror_to_node(seg_id, last_ras)
                    self._log.append({
                        'type':         'point',
                        'segment_id':   seg_id,
                        'coord_ras':    list(last_ras),
                        'timestamp':    last_ts,
                        'cp_id':        None,
                        'mirror_cp_id': mirror_cp_id,
                    })
                    count += 1
                else:
                    # Deleted point — no mirror node; use original key as history key
                    mirror_cp_id = _key or f'_deleted_{id(point)}'

                versions = [
                    {'coord_ras': r, 'timestamp': t}
                    for r, t in zip(ras_list, ts_list)
                ]
                self._point_history[mirror_cp_id] = {
                    'segment_id': seg_id,
                    'versions':   versions,
                    'alive':      alive,
                }
        return count

    def _load_flat(self, json_data: list) -> int:
        count = 0
        for entry in json_data:
            seg_id = entry.get('segment_id', '')
            ras    = entry.get('coord_ras', [0.0, 0.0, 0.0])
            ts     = entry.get('timestamp', '')
            if not seg_id:
                continue
            mirror_cp_id = self._mirror_to_node(seg_id, ras)
            self._log.append({
                'type':         'point',
                'segment_id':   seg_id,
                'coord_ras':    list(ras),
                'timestamp':    ts,
                'cp_id':        None,
                'mirror_cp_id': mirror_cp_id,
            })
            self._point_history[mirror_cp_id] = {
                'segment_id': seg_id,
                'versions':   [{'coord_ras': list(ras), 'timestamp': ts}],
                'alive':      True,
            }
            count += 1
        return count

    def on_import(self, widget) -> None:
        """Open an open-file dialog, load JSON, and replay the annotation log."""
        import json
        import qt
        path = qt.QFileDialog.getOpenFileName(
            None, 'Import Annotation Log', '', 'JSON files (*.json)'
        )
        if not path:
            return
        with open(path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, (dict, list)):
            import slicer
            slicer.util.errorDisplay('Invalid annotation log: expected a JSON object or array.')
            return
        n = self.load_from_json(data)
        import slicer
        slicer.util.infoDisplay(f'Imported {n} annotation point(s) from:\n{path}')

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mirror_to_node(self, segment_id: str, ras) -> str:
        """Add *ras* to the persistent per-segment mirror node; return its cp_id."""
        import slicer
        import vtk
        node = self._nodes.get(segment_id)
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLMarkupsFiducialNode',
                f'AnnotationLog_{segment_id}',
            )
            node.CreateDefaultDisplayNodes()
            color = self._color_for(segment_id)
            dn = node.GetDisplayNode()
            if dn:
                dn.SetSelectedColor(*color)
                dn.SetColor(*color)
            node.SetAttribute('AnnotationLog', '1')
            self._nodes[segment_id] = node
            node.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                lambda caller, event, sid=segment_id: self._on_mirror_point_moved(caller, sid),
            )
            node.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
                lambda caller, event, sid=segment_id: self._on_mirror_point_removed(caller, sid),
            )
        idx = node.AddControlPoint(vtk.vtkVector3d(ras[0], ras[1], ras[2]))
        return node.GetNthControlPointID(idx)

    def _on_mirror_point_moved(self, node, segment_id: str) -> None:
        """Append a new version and update the log entry when a mirror point is dragged."""
        from datetime import datetime
        ts = datetime.now().isoformat()
        for entry in self._log:
            if entry.get('type') != 'point' or entry.get('segment_id') != segment_id:
                continue
            mid = entry.get('mirror_cp_id')
            if not mid:
                continue
            idx = node.GetControlPointIndexByID(mid)
            if idx < 0:
                continue
            pos = list(node.GetNthControlPointPosition(idx))
            if pos != entry['coord_ras']:
                entry['coord_ras'] = pos
                entry['timestamp'] = ts
                if mid in self._point_history:
                    self._point_history[mid]['versions'].append(
                        {'coord_ras': pos, 'timestamp': ts}
                    )

    def _on_mirror_point_removed(self, node, segment_id: str) -> None:
        """Handle manual deletion of a mirror point: append None version, clean up log.

        For Ctrl+Z undo, on_point_undone updates history and removes the log entry
        *before* calling RemoveNthControlPoint, so this observer finds no orphaned
        entries in that case and is a no-op.
        """
        from datetime import datetime
        ts = datetime.now().isoformat()
        orphaned = [
            i for i, e in enumerate(self._log)
            if e.get('type') == 'point'
            and e.get('segment_id') == segment_id
            and node.GetControlPointIndexByID(e.get('mirror_cp_id', '')) < 0
        ]
        for i in reversed(orphaned):
            mid = self._log[i].get('mirror_cp_id')
            if mid and mid in self._point_history:
                self._point_history[mid]['versions'].append(
                    {'coord_ras': None, 'timestamp': ts}
                )
                self._point_history[mid]['alive'] = False
            del self._log[i]

    def _color_for(self, segment_id: str) -> tuple:
        """Return the palette color for *segment_id*, assigning one if new."""
        if segment_id not in self._seg_colors:
            idx = len(self._seg_colors) % len(self._PALETTE)
            self._seg_colors[segment_id] = self._PALETTE[idx]
        return self._seg_colors[segment_id]
