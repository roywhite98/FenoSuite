"""Uncertain main paths must stay reviewable across UI, rebuild and export."""
from pathlib import Path
import tempfile
import unittest

from plant_phenotype.root_phenotype.vector.editing import apply_edit
from plant_phenotype.root_phenotype.vector.io import load_document, save_document, export_svg
from plant_phenotype.root_phenotype.vector.metrics import measure_vectors
from plant_phenotype.root_phenotype.vector.models import RootNode, VectorRoot
from plant_phenotype.root_phenotype.vector.pipeline import analyze_vectors
from plant_phenotype.root_phenotype.vector.quality import _has_thin_detour
from plant_phenotype.root_phenotype.vector.resolve import rebuild_document


class PrimaryQualityTests(unittest.TestCase):
    def test_width_changes_require_long_thin_interval_between_thick_sections(self):
        profiles = [([10]*10+[4]*40+[10]*10, True),
                    ([10]*10+[4]*15+[10]*10, False),
                    ([10]*10+[4]*50, False),
                    ([10]*60, False)]
        for widths, expected in profiles:
            root = VectorRoot('r', [RootNode(str(i), i, 0, width)
                                    for i, width in enumerate(widths)])
            with self.subTest(widths=widths):
                self.assertEqual(_has_thin_detour(root), expected)
        # An unmeasured repair is not evidence of a thin side-root detour.
        root = VectorRoot('r', [RootNode(str(i), i, 0, width) for i, width in
                               enumerate([10]*10+[4]*40+[10]*10)])
        for node in root.nodes[10:50]:
            node.diameter_px, node.diameter_source = None, 'missing'
        self.assertFalse(_has_thin_detour(root))

    def test_combo_review_survives_rebuild_roundtrip_and_measurement_export(self):
        path = Path(__file__).parent/'fixtures/root_vector/combo_012.png'
        original = analyze_vectors(path, selection_points=[(167, 30)]).document
        document = rebuild_document(original)
        primary = next(r for r in document.roots if document.root_order(r.id) == 0)
        self.assertEqual(primary.review_state, 'pending')
        self.assertEqual(primary.quality_flags.count('possible_trunk_detour'), 1)
        self.assertEqual(sum(s.startswith('主根候选 ') for s in document.diagnostics), 1)
        for root in document.roots:
            if document.root_order(root.id) not in (0, None):
                self.assertEqual(root.review_state, 'pending')
                self.assertIn('uncertain_primary_ancestry', root.quality_flags)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'result.json'
            save_document(document, target)
            loaded = load_document(target)
            row = measure_vectors(loaded)['roots'].set_index('root_id').loc[primary.id]
            self.assertEqual(row['review_state'], 'pending')
            self.assertIn('possible_trunk_detour', row['quality_flags'])
            svg = Path(directory)/'result.svg'
            export_svg(loaded, svg)
            self.assertIn('stroke-dasharray', svg.read_text(encoding='utf-8'))
        confirmed = apply_edit(document, {'type': 'confirm_root', 'root_id': primary.id}).document
        rebuilt = rebuild_document(confirmed)
        self.assertEqual(next(r for r in rebuilt.roots if r.id == primary.id).review_state, 'confirmed')

