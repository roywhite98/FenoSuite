"""Synthetic, semantic checks for vector continuation and root ancestry."""
import copy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from plant_phenotype.root_phenotype.vector.editing import apply_edit, resolve_decisions
from plant_phenotype.root_phenotype.vector.features import extract_features, mask_digest
from plant_phenotype.root_phenotype.vector.models import VectorAnalysisConfig, validate_document
from plant_phenotype.root_phenotype.vector.pipeline import analyze_vectors
from plant_phenotype.root_phenotype.vector.resolve import rebuild_document


def branch_mask():
    mask = np.zeros((121, 121), np.uint8)
    cv2.line(mask, (60, 10), (60, 110), 255, 5)
    cv2.line(mask, (60, 60), (100, 100), 255, 3)
    return mask


def cross_mask():
    mask = np.zeros((121, 121), np.uint8)
    cv2.line(mask, (60, 10), (60, 110), 255, 3)
    cv2.line(mask, (10, 60), (110, 60), 255, 3)
    return mask


class VectorAlgorithmTests(unittest.TestCase):
    def test_tensor_is_tangent_and_cross_center_is_invalid(self):
        mask = np.zeros((101, 101), np.uint8)
        mask[10:91, 48:53] = 255
        vertical = extract_features(mask, VectorAnalysisConfig())
        self.assertTrue(vertical.field_valid[50, 50])
        self.assertLess(vertical.cos2[50, 50], -.99)
        horizontal = extract_features(mask.T, VectorAnalysisConfig())
        self.assertGreater(horizontal.cos2[50, 50], .99)
        cross = extract_features(cross_mask(), VectorAnalysisConfig())
        self.assertFalse(cross.field_valid[60, 60])

    def test_thick_interior_without_edges_has_no_fake_direction(self):
        mask = np.zeros((301, 301), np.uint8)
        mask[10:291, 70:231] = 255
        features = extract_features(mask, VectorAnalysisConfig())
        self.assertFalse(features.field_valid[150, 150])
        self.assertTrue(np.isfinite(features.diameter).all())

    def test_staircase_does_not_create_spurious_junctions(self):
        mask = np.zeros((100, 100), np.uint8)
        for x in range(10, 85):
            mask[x//2+10, x] = 255
        doc = analyze_vectors(mask).document
        self.assertEqual(len(doc.analysis_snapshot.segments), 1)
        self.assertEqual(doc.junction_decisions, [])
        self.assertIsNone(doc.root_order(doc.roots[0].id))

    def test_clear_branch_builds_primary_and_lateral_for_all_rotations(self):
        mask, base = branch_mask(), (60, 10)
        for _ in range(4):
            doc = analyze_vectors(mask, [base]).document
            self.assertEqual(sorted(doc.root_order(r.id) for r in doc.roots), [0, 1])
            self.assertEqual(len(doc.analysis_snapshot.segments), 3)
            child = next(r for r in doc.roots if r.parent_attachment)
            parent = doc.root_by_id(child.parent_attachment.parent_root_id)
            node = next(n for n in parent.nodes if n.id == child.parent_attachment.parent_node_id)
            self.assertEqual((node.x, node.y), (child.nodes[0].x, child.nodes[0].y))
            validate_document(doc)
            base = (base[1], mask.shape[1]-1-base[0])
            mask = np.rot90(mask)

    def test_higher_order_is_recursive_parentage(self):
        mask = np.zeros((180, 180), np.uint8)
        for a, b, width in [((40, 10), (40, 160), 5), ((40, 60), (115, 135), 3), ((80, 100), (140, 80), 2)]:
            cv2.line(mask, a, b, 255, width)
        doc = analyze_vectors(mask, [(40, 10)]).document
        self.assertEqual(sorted(doc.root_order(r.id) for r in doc.roots), [0, 1, 2])

    def test_symmetric_y_stays_pending_instead_of_longest_path(self):
        mask = np.zeros((121, 121), np.uint8)
        for a, b in [((60, 10), (60, 60)), ((60, 60), (20, 100)), ((60, 60), (100, 100))]:
            cv2.line(mask, a, b, 255, 3)
        doc = analyze_vectors(mask, [(60, 10)]).document
        self.assertEqual(doc.junction_decisions[0].state, 'pending')
        self.assertFalse(any(r.parent_attachment for r in doc.roots))
        self.assertEqual(sum(doc.root_order(r.id) == 0 for r in doc.roots), 1)

    def test_perpendicular_touch_is_not_forced_into_parentage(self):
        mask = np.zeros((100, 100), np.uint8)
        cv2.line(mask, (40, 10), (40, 90), 255, 3)
        cv2.line(mask, (40, 50), (90, 50), 255, 3)
        doc = analyze_vectors(mask, [(40, 10)]).document
        self.assertEqual(doc.junction_decisions[0].state, 'pending')

    def test_confirmed_crossing_connects_two_roots_without_parentage(self):
        doc = analyze_vectors(cross_mask(), [(60, 10)]).document
        before = copy.deepcopy(doc.to_dict())
        decision = doc.junction_decisions[0]
        self.assertEqual(decision.state, 'pending')
        crossing = min((a for a in decision.alternatives if a.kind == 'crossing'), key=lambda a: a.cost)
        updated = resolve_decisions(doc, {decision.id: crossing.id}).document
        self.assertEqual(doc.to_dict(), before)
        self.assertEqual(len(updated.roots), 2)
        self.assertTrue(all(len(r.source_segment_ids) == 2 for r in updated.roots))
        self.assertFalse(any(r.parent_attachment for r in updated.roots))
        self.assertEqual(sum(updated.root_order(r.id) == 0 for r in updated.roots), 1)

    def test_cycle_is_kept_and_terminates(self):
        mask = np.zeros((121, 121), np.uint8)
        cv2.circle(mask, (60, 60), 35, 255, 2)
        doc = analyze_vectors(mask).document
        self.assertEqual(len(doc.roots), 1)
        self.assertTrue(doc.analysis_snapshot.segments[0].closed)
        root = doc.roots[0]
        self.assertEqual(root.completeness, 'loop')
        self.assertEqual((root.nodes[0].x, root.nodes[0].y), (root.nodes[-1].x, root.nodes[-1].y))
        self.assertNotEqual(root.nodes[0].id, root.nodes[-1].id)

    def test_no_base_does_not_implicitly_choose_a_primary(self):
        doc = analyze_vectors(branch_mask()).document
        self.assertTrue(all(doc.root_order(r.id) is None for r in doc.roots))
        self.assertFalse(any(r.parent_attachment for r in doc.roots))

    def test_middle_click_and_competing_bases_are_unresolved(self):
        mask = np.zeros((121, 121), np.uint8)
        cv2.line(mask, (60, 10), (60, 110), 255, 3)
        middle = analyze_vectors(mask, [(60, 60)]).document
        self.assertIsNone(middle.base_marks[0].root_id)
        both = analyze_vectors(mask, [(60, 10), (60, 110)]).document
        self.assertTrue(all(b.root_id is None for b in both.base_marks))
        self.assertTrue(all(both.root_order(r.id) is None for r in both.roots))
        self.assertTrue(any('争用' in s for s in both.diagnostics))

    def test_rebuild_preserves_manual_node_and_does_not_mutate_snapshot(self):
        doc = analyze_vectors(branch_mask(), [(60, 10)]).document
        root = next(r for r in doc.roots if r.base_mark_id)
        node = root.nodes[1]
        edited = apply_edit(doc, {'type': 'move_node', 'root_id': root.id, 'node_id': node.id, 'x': node.x+1, 'y': node.y}).document
        snapshot = copy.deepcopy(edited.analysis_snapshot)
        rebuilt = rebuild_document(edited)
        point = next(n for n in rebuilt.root_by_id(root.id).nodes if n.id == node.id)
        self.assertEqual(point.x, node.x+1)
        self.assertIn('manual_locked', point.tags)
        self.assertEqual(edited.analysis_snapshot, snapshot)

    def test_deleted_segments_are_not_resurrected(self):
        doc = analyze_vectors(branch_mask(), [(60, 10)]).document
        child = next(r for r in doc.roots if r.parent_attachment)
        edited = apply_edit(doc, {'type': 'delete_root', 'root_id': child.id}).document
        rebuilt = rebuild_document(edited)
        self.assertFalse(any(set(child.source_segment_ids) & set(r.source_segment_ids) for r in rebuilt.roots))

    def test_deterministic_identifiers_and_binary_source_hash(self):
        mask = branch_mask()
        a = analyze_vectors(mask, [(60, 10)]).document
        b = analyze_vectors(mask > 0, [(60, 10)]).document
        self.assertEqual(a.analysis_snapshot.analysis_id, b.analysis_snapshot.analysis_id)
        self.assertEqual([r.id for r in a.roots], [r.id for r in b.roots])
        self.assertEqual(mask_digest(mask), mask_digest(mask > 0))

    def test_all_components_and_small_real_roots_are_retained(self):
        mask = branch_mask()
        mask[100:104, 10] = 255
        doc = analyze_vectors(mask).document
        self.assertEqual(len(doc.analysis_snapshot.segments), 4)

    def test_blank_is_valid_and_isolated_pixel_is_reported(self):
        mask = np.zeros((20, 20), np.uint8)
        blank = analyze_vectors(mask).document
        self.assertEqual(blank.status, 'empty')
        mask[10, 10] = 255
        isolated = analyze_vectors(mask).document
        self.assertFalse(isolated.roots)
        self.assertTrue(any('孤立像素' in s for s in isolated.diagnostics))

    def test_progress_cancel_and_mismatched_original(self):
        events = []
        analyze_vectors(branch_mask(), progress=lambda current, total, message: events.append((current, total, message)))
        self.assertEqual(events[-1][0:2], (100, 100))
        def cancel():
            raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            analyze_vectors(branch_mask(), check_cancel=cancel)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'original.png'
            cv2.imencode('.png', np.zeros((30, 30), np.uint8))[1].tofile(path)
            with self.assertRaisesRegex(ValueError, '尺寸'):
                analyze_vectors(branch_mask(), original_image_path=path)


if __name__ == '__main__':
    unittest.main()
