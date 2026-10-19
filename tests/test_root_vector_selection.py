"""End-to-end contracts for arbitrary in-root clicks and scoped root analysis."""
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from plant_phenotype.root_phenotype.vector.pipeline import analyze_vectors
from plant_phenotype.root_phenotype.vector.selection import prune_spurs
from plant_phenotype.root_phenotype.vector.io import load_document, save_document, root_color


def specimen():
    mask = np.zeros((200, 220), np.uint8)
    cv2.line(mask, (65, 15), (65, 180), 255, 7)
    cv2.line(mask, (65, 70), (115, 100), 255, 3)
    cv2.line(mask, (65, 125), (20, 155), 255, 3)
    # An unselected, parallel plant must not leak into the output.
    cv2.line(mask, (190, 15), (190, 180), 255, 7)
    return mask


class SelectedRootTests(unittest.TestCase):
    def test_base_middle_primary_tip_and_lateral_tip_choose_same_whole_root(self):
        shapes = []
        for point in [(65, 15), (65, 95), (65, 180), (115, 100)]:
            result = analyze_vectors(specimen(), selection_points=[point])
            doc = result.document
            primary = [r for r in doc.roots if doc.root_order(r.id) == 0]
            lateral = [r for r in doc.roots if doc.root_order(r.id) == 1]
            self.assertEqual(len(primary), 1)
            self.assertEqual(len(lateral), 2)
            self.assertLess(min(n.y for n in primary[0].nodes), 20)
            self.assertGreater(max(n.y for n in primary[0].nodes), 175)
            self.assertFalse(result.mask[:, 185:].any())
            self.assertTrue(all(n.x < 130 for r in doc.roots for n in r.nodes))
            self.assertNotEqual(root_color(primary[0].id, doc), root_color(lateral[0].id, doc))
            shapes.append([(doc.root_order(r.id), [(n.x, n.y) for n in r.nodes]) for r in doc.roots])
        self.assertTrue(all(s == shapes[0] for s in shapes))

    def test_broken_root_is_recovered_from_either_fragment(self):
        mask = specimen()
        mask[102:110, 59:72] = 0
        for point in [(65, 30), (65, 170)]:
            result = analyze_vectors(mask, selection_points=[point])
            doc = result.document
            primary = [r for r in doc.roots if doc.root_order(r.id) == 0]
            self.assertEqual(len(primary), 1)
            self.assertGreater(max(n.y for n in primary[0].nodes)-min(n.y for n in primary[0].nodes), 155)
            self.assertTrue(result.mask[105, 65])
            self.assertFalse(result.mask[:, 185:].any())
            repaired = [n for r in doc.roots for n in r.nodes if 'repaired' in n.tags]
            self.assertTrue(repaired)
            self.assertTrue(all(n.diameter_px is None for n in repaired))
        isolated = analyze_vectors(mask, selection_points=[(65, 30)], config={'connect_gap_px': 0})
        self.assertFalse(isolated.mask[170, 65])

    def test_multiple_clicks_on_same_plant_deduplicate_and_other_plant_is_explicit(self):
        doc = analyze_vectors(specimen(), selection_points=[(65, 30), (115, 100)]).document
        self.assertEqual(sum(doc.root_order(r.id) == 0 for r in doc.roots), 1)
        doc = analyze_vectors(specimen(), selection_points=[(65, 30), (190, 80)]).document
        self.assertEqual(sum(doc.root_order(r.id) == 0 for r in doc.roots), 2)

    def test_empty_or_background_selection_never_analyzes_entire_image(self):
        for points in ([], [(0, 0)], [(-1, 40)], [(float('nan'), 20)]):
            with self.assertRaises(ValueError):
                analyze_vectors(specimen(), selection_points=points)

    def test_short_spurs_removed_but_real_branches_and_standalone_roots_survive(self):
        skel = np.zeros((50, 50), bool)
        skel[5:45, 20] = True
        skel[20, 20:24] = True
        skel[30, 20:36] = True
        skel[4:8, 40] = True
        cleaned, count = prune_spurs(skel, 5)
        self.assertEqual(count, 3)
        self.assertFalse(cleaned[20, 21:24].any())
        self.assertTrue(cleaned[30, 20:36].all())
        self.assertTrue(cleaned[4:8, 40].all())

    def test_rotated_root_keeps_same_primary_and_secondary_branch_hierarchy(self):
        mask = specimen()
        cv2.line(mask, (95, 88), (128, 65), 255, 1)
        for k in range(4):
            rotated = np.rot90(mask, k).copy()
            seed_mask = np.zeros_like(mask)
            seed_mask[70, 65] = 1
            row, col = np.argwhere(np.rot90(seed_mask, k))[0]
            doc = analyze_vectors(rotated, selection_points=[(int(col), int(row))]).document
            self.assertEqual(sorted(doc.root_order(r.id) for r in doc.roots), [0, 1, 1, 2])

    def test_more_than_three_breaks_reconnect_from_either_end(self):
        mask = np.zeros((200, 80), np.uint8)
        cv2.line(mask, (40, 10), (40, 190), 255, 5)
        for y in (35, 65, 95, 125, 155):
            mask[y:y+6, :] = 0
        for seed in [(40, 10), (40, 190)]:
            result = analyze_vectors(mask, selection_points=[seed])
            self.assertEqual(len(result.document.roots), 1)
            self.assertTrue(result.mask[40:185, 40].all())

    def test_nearby_parallel_root_is_not_connected_sideways(self):
        mask = np.zeros((150, 100), np.uint8)
        cv2.line(mask, (40, 10), (40, 130), 255, 3)
        cv2.line(mask, (50, 10), (50, 130), 255, 3)
        result = analyze_vectors(mask, selection_points=[(40, 70)])
        self.assertFalse(result.mask[:, 48:].any())
        self.assertEqual(len(result.document.roots), 1)

    def test_thick_cut_ends_use_background_gap_not_retracted_skeleton_distance(self):
        mask = np.zeros((240, 100), np.uint8)
        cv2.line(mask, (40, 10), (40, 225), 255, 11)
        mask[100:112, :] = 0
        result = analyze_vectors(mask, selection_points=[(40, 20)], config={'connect_gap_px': 15})
        self.assertEqual(len(result.document.roots), 1)
        self.assertTrue(result.mask[220, 40])
        self.assertTrue(result.mask[106, 40])

    def test_selection_and_automatic_ancestry_survive_json_round_trip(self):
        doc = analyze_vectors(specimen(), selection_points=[(115, 100)]).document
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'roots.json'
            save_document(doc, path)
            restored = load_document(path)
        self.assertEqual(restored.selection_points, doc.selection_points)
        self.assertEqual([restored.root_order(r.id) for r in restored.roots], [0, 1, 1])


if __name__ == '__main__':
    unittest.main()
