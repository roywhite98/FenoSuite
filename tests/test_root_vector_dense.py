"""Main-axis regression cases with dense side roots near the distal end."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from plant_phenotype.root_phenotype.vector.pipeline import analyze_vectors


def dense_distal_roots(step=10, trunk_end=1000, tail_span=200):
    """A gently bending thick axis with thin, first-order lateral roots."""
    mask = np.zeros((220, trunk_end + tail_span + 50), np.uint8)
    cv2.line(mask, (10, 100), (trunk_end, 100), 255, 9)
    tail = [(trunk_end, 100)] + [
        (trunk_end + offset, 100 + (3 if i % 2 else -3))
        for i, offset in enumerate(range(20, tail_span + 20, 20))
    ]
    cv2.polylines(mask, [np.array(tail)], False, 255, 9)
    cv2.line(mask, (trunk_end, 100), (trunk_end, 125), 255, 3)
    for i, offset in enumerate(range(30, tail_span, step)):
        x = trunk_end + offset
        cv2.line(mask, (x, 100), (x + (8 if i % 2 else -8), 70 if i % 2 else 130), 255, 2)
    return mask


def rotate_point(point, shape, turns):
    x, y = point
    height, width = shape
    for _ in range(turns):
        x, y = y, width - 1 - x
        height, width = width, height
    return x, y


class DenseDistalRootTests(unittest.TestCase):
    def assert_complete_axis(self, document, start, end):
        primary = [r for r in document.roots if document.root_order(r.id) == 0]
        self.assertEqual(len(primary), 1)
        endpoints = [(primary[0].nodes[0].x, primary[0].nodes[0].y),
                     (primary[0].nodes[-1].x, primary[0].nodes[-1].y)]
        # Direction can legitimately flip when the two ends have equal width.
        for expected in (start, end):
            self.assertLessEqual(
                min(np.linalg.norm(np.subtract(actual, expected)) for actual in endpoints), 4,
                f"Main root ended at {endpoints}, missing thick-axis endpoint {expected}",
            )
        return primary[0]

    def assert_first_order_laterals(self, document, primary):
        lateral = [r for r in document.roots if r.id != primary.id]
        self.assertTrue(lateral)
        self.assertTrue(
            all(document.root_order(r.id) == 1 for r in lateral),
            "All drawn side roots attach directly to the primary axis; none is second order",
        )
        self.assertTrue(all(r.parent_attachment.parent_root_id == primary.id for r in lateral))

    def test_increasing_lateral_density_does_not_redirect_main_axis(self):
        for step in (25, 20, 15, 10):
            with self.subTest(branch_spacing=step):
                document = analyze_vectors(dense_distal_roots(step), selection_points=[(500, 100)]).document
                primary = self.assert_complete_axis(document, (10, 100), (1200, 103))
                self.assert_first_order_laterals(document, primary)
                self.assertEqual(primary.review_state, 'auto')

    def test_dense_laterals_keep_primary_axis_and_order_after_rotation(self):
        mask = dense_distal_roots()
        for turns in range(4):
            with self.subTest(quarter_turns=turns):
                seed = rotate_point((500, 100), mask.shape, turns)
                document = analyze_vectors(np.rot90(mask, turns).copy(), selection_points=[seed]).document
                primary = self.assert_complete_axis(
                    document,
                    rotate_point((10, 100), mask.shape, turns),
                    rotate_point((1200, 103), mask.shape, turns),
                )
                self.assert_first_order_laterals(document, primary)

    def test_clicking_primary_or_lateral_does_not_change_dense_root_result(self):
        mask = dense_distal_roots()
        signatures = []
        for seed in ((10, 100), (500, 100), (1200, 103), (1022, 130)):
            with self.subTest(seed=seed):
                document = analyze_vectors(mask, selection_points=[seed]).document
                primary = self.assert_complete_axis(document, (10, 100), (1200, 103))
                self.assert_first_order_laterals(document, primary)
                signatures.append([
                    (document.root_order(r.id), [(n.x, n.y) for n in r.nodes])
                    for r in document.roots
                ])
        self.assertTrue(all(signature == signatures[0] for signature in signatures))

    def test_lengthening_shared_trunk_does_not_change_distal_continuation(self):
        for trunk_end in (200, 500, 1000):
            with self.subTest(trunk_end=trunk_end):
                document = analyze_vectors(
                    dense_distal_roots(trunk_end=trunk_end), selection_points=[(50, 100)],
                ).document
                primary = self.assert_complete_axis(document, (10, 100), (trunk_end + 200, 103))
                self.assert_first_order_laterals(document, primary)

    def test_more_than_24_tips_preserve_axis_and_all_first_order_laterals(self):
        mask = dense_distal_roots(tail_span=400)
        for turns in range(4):
            with self.subTest(quarter_turns=turns):
                document = analyze_vectors(
                    np.rot90(mask, turns).copy(),
                    selection_points=[rotate_point((500, 100), mask.shape, turns)],
                ).document
                primary = self.assert_complete_axis(
                    document,
                    rotate_point((10, 100), mask.shape, turns),
                    rotate_point((1400, 103), mask.shape, turns),
                )
                # 37 alternating laterals plus the short branch at the tail base.
                self.assertEqual(len(document.roots), 39)
                self.assert_first_order_laterals(document, primary)

    def test_long_thin_lateral_does_not_replace_thick_continuous_axis(self):
        mask = np.zeros((1000, 500), np.uint8)
        cv2.line(mask, (50, 150), (450, 150), 255, 9)
        cv2.line(mask, (250, 150), (250, 950), 255, 2)
        document = analyze_vectors(mask, selection_points=[(250, 900)]).document
        primary = self.assert_complete_axis(document, (50, 150), (450, 150))
        self.assertEqual(len(document.roots), 2)
        self.assert_first_order_laterals(document, primary)

    def test_short_connector_at_bent_four_way_fork_does_not_lock_wrong_axis(self):
        mask = np.zeros((340, 500), np.uint8)
        cv2.line(mask, (20, 160), (250, 160), 255, 9)
        cv2.line(mask, (250, 160), (360, 56), 255, 9)
        cv2.line(mask, (250, 160), (250, 25), 255, 7)
        cv2.line(mask, (250, 160), (250, 300), 255, 7)
        # Skeletonization splits the four-way fork into two adjacent junctions.
        # Their tiny connecting segment cannot establish a reliable direction.
        document = analyze_vectors(
            mask, selection_points=[(30, 160)],
            config={'connect_gap_px': 0, 'spur_length_px': 0},
        ).document
        primary = self.assert_complete_axis(document, (20, 160), (360, 56))
        self.assertEqual(len(document.roots), 3)
        self.assert_first_order_laterals(document, primary)

    def test_straight_side_branch_does_not_truncate_bending_axis(self):
        mask = np.zeros((820, 500), np.uint8)
        cv2.polylines(mask, [np.array([(300, 20), (300, 200), (140, 360),
                                      (220, 555), (300, 750)])], False, 255, 7)
        cv2.line(mask, (300, 200), (300, 330), 255, 7)
        cv2.line(mask, (220, 555), (140, 680), 255, 5)
        # Equal-width roots give no independent evidence to lock the straight
        # short branch. The complete curved axis must beat that local pair.
        for turns in range(4):
            for seed in ((300, 50), (300, 320), (140, 680)):
                with self.subTest(turns=turns, seed=seed):
                    document = analyze_vectors(
                        np.rot90(mask, turns).copy(),
                        selection_points=[rotate_point(seed, mask.shape, turns)],
                    ).document
                    primary = self.assert_complete_axis(
                        document, rotate_point((300, 20), mask.shape, turns),
                        rotate_point((300, 750), mask.shape, turns),
                    )
                    self.assertEqual(len(document.roots), 3)
                    self.assert_first_order_laterals(document, primary)

    def test_thin_loop_shortcut_does_not_replace_curved_main_segment(self):
        mask = np.zeros((360, 240), np.uint8)
        cv2.polylines(mask, [np.array([(120, 10), (120, 40), (50, 130),
                                      (120, 220), (120, 330)])], False, 255, 9)
        cv2.line(mask, (120, 40), (120, 220), 255, 2)
        for turns in range(4):
            with self.subTest(turns=turns):
                document = analyze_vectors(
                    np.rot90(mask, turns).copy(),
                    selection_points=[rotate_point((120, 20), mask.shape, turns)],
                ).document
                primary = self.assert_complete_axis(
                    document, rotate_point((120, 10), mask.shape, turns),
                    rotate_point((120, 330), mask.shape, turns),
                )
                bend = rotate_point((50, 130), mask.shape, turns)
                self.assertLess(min(np.linalg.norm(np.subtract((n.x, n.y), bend))
                                    for n in primary.nodes), 5)
                self.assertNotIn('possible_trunk_detour', primary.quality_flags)
                self.assertTrue(any('闭环' in note for note in document.diagnostics))

    def test_combo012_thick_base_is_not_replaced_by_lateral_endpoint(self):
        # The original mask contains missing main-root sections and thin
        # detours. This asserts endpoint identity only, not full gap recovery.
        path = Path(__file__).parent / 'fixtures/root_vector/combo_012.png'
        mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        for turns, seed in ((0, (167, 30)), (1, (167, 30)),
                            (2, (167, 30)), (3, (167, 30)),
                            (0, (210, 493)), (0, (137, 309))):
            with self.subTest(turns=turns, seed=seed):
                document = analyze_vectors(
                    np.rot90(mask, turns).copy(),
                    selection_points=[rotate_point(seed, mask.shape, turns)],
                ).document
                primary = self.assert_complete_axis(
                    document, rotate_point((166, 29), mask.shape, turns),
                    rotate_point((210, 493), mask.shape, turns),
                )
                self.assertTrue(any('绕过了主干断口' in note for note in document.diagnostics))
                self.assertEqual(primary.review_state, 'pending')
                self.assertIn('possible_trunk_detour', primary.quality_flags)


if __name__ == '__main__':
    unittest.main()
