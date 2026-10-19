"""Vector measurement and interchange contracts, including malformed input."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plant_phenotype.root_phenotype.vector.io import (
    export_svg, export_tables, load_document, root_color, save_document,
)
from plant_phenotype.root_phenotype.vector.metrics import measure_vectors
from plant_phenotype.root_phenotype.vector.models import (
    BaseMark, ParentAttachment, RootDocument, RootNode, SourceRef, VectorRoot,
)


def example_document():
    return RootDocument(
        source=SourceRef(width=100, height=100, mm_per_pixel=0.2),
        base_marks=[BaseMark('base', 10, 0, root_id='main', specimen_id='样本一')],
        roots=[
            VectorRoot('main', [RootNode('a', 10, 0, 2), RootNode('b', 10, 10, 4),
                                RootNode('c', 10, 30, 6)],
                       base_mark_id='base', direction='base_to_tip', review_state='confirmed'),
            VectorRoot('lateral', [RootNode('d', 10, 10, 2), RootNode('e', 20, 10, 2)],
                       parent_attachment=ParentAttachment('main', 'b'),
                       direction='base_to_tip', review_state='auto'),
            VectorRoot('unassigned', [RootNode('f', 40, 40, None, 'missing'),
                                      RootNode('g', 45, 40, None, 'missing')]),
        ],
    )


class TestVectorMeasurements(unittest.TestCase):
    def test_lengths_weighted_diameter_ancestry_and_calibration(self):
        tables = measure_vectors(example_document())
        roots = tables['roots'].set_index('root_id')
        self.assertEqual(set(tables), {'roots', 'nodes', 'attachments', 'summary'})
        self.assertEqual(roots.loc['main', 'length_px'], 30)
        self.assertAlmostEqual(roots.loc['main', 'mean_diameter_px'], 13 / 3)
        self.assertEqual(roots.loc['main', 'length_mm'], 6)
        self.assertEqual(roots.loc['lateral', 'order'], 1)
        self.assertTrue(pd.isna(roots.loc['unassigned', 'order']))
        self.assertTrue(pd.isna(roots.loc['unassigned', 'mean_diameter_px']))
        self.assertFalse(roots.loc['unassigned', 'is_visible_tip'])
        summary = tables['summary'].iloc[0]
        self.assertEqual(summary.root_count, 2)
        self.assertEqual(summary.unassigned_count, 1)
        self.assertEqual(summary.total_length_px, 45)

    def test_diameter_coverage_excludes_interpolated_intervals(self):
        document = example_document()
        document.roots[0].nodes[2].diameter_source = 'interpolated'
        root = measure_vectors(document)['roots'].iloc[0]
        self.assertAlmostEqual(root.diameter_coverage, 1 / 3)
        self.assertEqual(root.mean_diameter_px, 3)

    def test_angles_follow_parent_direction_and_attachment_arclength(self):
        document = example_document()
        # An upward child grows opposite to the parent, so its directed angle
        # must not be silently collapsed into the acute-angle convention.
        document.roots[1].nodes[-1].x = 10
        document.roots[1].nodes[-1].y = 0
        attachment = measure_vectors(document)['attachments'].iloc[0]
        self.assertAlmostEqual(attachment.branch_angle_deg, 180)
        self.assertAlmostEqual(attachment.acute_angle_deg, 0)
        self.assertEqual(attachment.attachment_s_px, 10)
        self.assertAlmostEqual(attachment.attachment_fraction, 1 / 3)
        document.roots[0].direction = 'unknown'
        attachment = measure_vectors(document)['attachments'].iloc[0]
        self.assertTrue(pd.isna(attachment.branch_angle_deg))
        self.assertAlmostEqual(attachment.acute_angle_deg, 0)

    def test_empty_and_uncalibrated_documents_do_not_invent_measurements(self):
        document = example_document()
        document.source.mm_per_pixel = None
        self.assertTrue(measure_vectors(document)['roots']['length_mm'].isna().all())
        tables = measure_vectors(RootDocument())
        self.assertTrue(tables['roots'].empty)
        self.assertEqual(tables['summary'].iloc[0].root_count, 0)


class TestVectorPersistence(unittest.TestCase):
    def test_algorithm_document_and_manual_geometry_round_trip(self):
        from plant_phenotype.root_phenotype.vector.editing import apply_edit, resolve_decisions
        from plant_phenotype.root_phenotype.vector.pipeline import analyze_vectors

        mask = np.zeros((121, 121), np.uint8)
        cv2.line(mask, (60, 10), (60, 110), 255, 5)
        cv2.line(mask, (60, 60), (100, 100), 255, 3)
        document = analyze_vectors(mask, [(60, 10)]).document
        main = next(root for root in document.roots if document.root_order(root.id) == 0)
        lateral = next(root for root in document.roots if document.root_order(root.id) == 1)
        node = next(node for node in main.nodes[1:-1] if 'attachment' not in node.tags)
        moved_x, node_id, main_id = node.x + 1, node.id, main.id
        changed = apply_edit(document, {'type': 'move_node', 'root_id': main_id,
                                       'node_id': node_id, 'x': moved_x, 'y': node.y}).document
        rebuilt = resolve_decisions(changed, {}).document
        self.assertEqual(next(n for n in rebuilt.root_by_id(main_id).nodes if n.id == node_id).x, moved_x)
        self.assertNotEqual(next(n for n in document.root_by_id(main_id).nodes if n.id == node_id).x, moved_x)
        deleted = apply_edit(rebuilt, {'type': 'delete_root', 'root_id': lateral.id}).document
        rebuilt = resolve_decisions(deleted, {}).document
        self.assertNotIn(lateral.id, {root.id for root in rebuilt.roots})
        assigned = {segment for root in rebuilt.roots for segment in root.source_segment_ids}
        self.assertFalse(assigned.intersection(rebuilt.excluded_segment_ids))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / '矢量结果.json'
            save_document(rebuilt, path)
            loaded = load_document(path)
            self.assertEqual(next(n for n in loaded.root_by_id(main_id).nodes if n.id == node_id).x, moved_x)
            self.assertEqual(loaded.excluded_segment_ids, rebuilt.excluded_segment_ids)

    def test_round_trip_unicode_and_relative_source_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / '根系数据'
            directory.mkdir()
            document = example_document()
            document.source.mask_path = str(directory / '根掩码.png')
            original = document.to_dict()
            path = directory / '结果.json'
            save_document(document, path)
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['source']['mask_path'], '根掩码.png')
            loaded = load_document(path)
            self.assertEqual(loaded.to_dict(), original)
            self.assertEqual(document.to_dict(), original, 'saving must not mutate the active document')

    def test_invalid_values_schema_references_and_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.json'
            mutations = [
                lambda p: p.update(schema_version=2),
                lambda p: p.update(revision=True),
                lambda p: p['source'].update(width=2.5),
                lambda p: p['roots'][0]['nodes'][0].update(x=float('nan')),
                lambda p: p['roots'][0]['nodes'][0].update(x=True),
                lambda p: p['roots'][0]['nodes'][0].update(diameter_px=-1),
                lambda p: p['roots'][0].update(completeness='imaginary'),
                lambda p: p['roots'][0].update(source_segment_ids=['missing']),
                lambda p: p['roots'][1]['parent_attachment'].update(parent_node_id='missing'),
                lambda p: p.update(unrecognized_field=10),
            ]
            for mutate in mutations:
                with self.subTest(mutate=mutate):
                    payload = example_document().to_dict()
                    mutate(payload)
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_document(path)
            path.write_text('{"schema_version": 1, "schema_version": 1}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '重复'):
                load_document(path)

    def test_atomic_failure_leaves_previous_file_and_no_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'document.json'
            path.write_text('previous', encoding='utf-8')
            with patch('plant_phenotype.root_phenotype.vector.io.os.replace', side_effect=OSError('blocked')):
                with self.assertRaises(OSError):
                    save_document(example_document(), path)
            self.assertEqual(path.read_text(encoding='utf-8'), 'previous')
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_exports_tables_formula_protection_and_svg_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            document = example_document()
            document.base_marks[0].specimen_id = '=1+1'
            csv_path, xlsx_path, svg_path = (directory / name for name in ('根.csv', '根.xlsx', '根.svg'))
            export_tables(document, csv_path)
            self.assertEqual(pd.read_csv(csv_path).iloc[0].specimen_id, "'=1+1")
            export_tables(document, xlsx_path)
            book = pd.ExcelFile(xlsx_path)
            self.assertEqual(book.sheet_names, ['roots', 'nodes', 'attachments', 'summary'])
            self.assertEqual(pd.read_excel(book, sheet_name='roots').iloc[0].specimen_id, "'=1+1")
            book.close()
            export_svg(document, svg_path)
            svg = ET.parse(svg_path).getroot()
            paths = svg.findall('.//{http://www.w3.org/2000/svg}polyline')
            self.assertEqual(len(paths), 3)
            self.assertEqual(paths[0].attrib['points'], '10,0 10,10 10,30')
            self.assertEqual(paths[0].attrib['stroke'], root_color('main', document))
            self.assertRegex(root_color('main'), '^#[0-9a-f]{6}$')


if __name__ == '__main__':
    unittest.main()
