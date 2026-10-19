"""核心根系算法审查缺陷的合成回归，不依赖真实图片或外部模型。"""
import itertools
import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plant_phenotype.root_phenotype.core import connect_roots, main_root, pipeline_stages, skeleton, stats
from plant_phenotype.root_phenotype.core.phenotyping_config import PhenotypingConfig
from plant_phenotype.root_phenotype.core.skeleton_cache import clear_skeleton_cache
from plant_phenotype.root_phenotype.facade import PhenotypingFacade


def _edge_key(u, v):
    return min(u, v), max(u, v)


class TestRootAlgorithmRegressions(unittest.TestCase):
    def setUp(self):
        clear_skeleton_cache()
        self.facade = PhenotypingFacade()
        self.config = {
            "spur_max_length": 0,
            "min_root_length": 0,
            "min_root_diameter": 0,
            "primary_root_enabled": False,
            "zscore": False,
            "connect_broken_roots": False,
        }

    def tearDown(self):
        clear_skeleton_cache()

    def _analyze(self, mask, seeds=None, connect=False):
        result = self.facade.analyze(
            mask,
            seeds=seeds,
            mode="interactive" if seeds else "batch",
            config_overrides={**self.config, "connect_broken_roots": connect, "connect_max_search": 20},
        )
        self.assertIsNone(result.error)
        return result

    def test_tree_diameter_uses_weights_instead_of_hop_count(self):
        center = (0, 0)
        graph = nx.Graph()
        for sign in (-1, 1):
            previous = center
            for step in (1, 2, 3):
                node = (0, sign * step)
                graph.add_edge(previous, node, weight=1.0)
                previous = node
        heavy_tip = (1, 0)
        graph.add_edge(center, heavy_tip, weight=100.0)
        path = main_root.find_longest_path_approximate(graph)
        self.assertIn(heavy_tip, (path[0], path[-1]))
        self.assertEqual(sum(graph[u][v]["weight"] for u, v in zip(path, path[1:])), 103.0)

    def test_tree_diameter_matches_all_pairs_weighted_reference(self):
        # 不同树形及权重验证双遍历，不把实现本身作为预期值。
        random = np.random.default_rng(42)
        for size in (2, 5, 20):
            graph = nx.Graph()
            for index in range(1, size):
                parent = int(random.integers(index))
                graph.add_edge((index, 0), (parent, 0), weight=float(random.uniform(0.1, 100)))
            path = main_root.find_longest_path_approximate(graph)
            actual = sum(graph[u][v]["weight"] for u, v in zip(path, path[1:]))
            expected = max(
                distance
                for distances in dict(nx.all_pairs_dijkstra_path_length(graph)).values()
                for distance in distances.values()
            )
            self.assertAlmostEqual(actual, expected)

    def test_array_analysis_does_not_reuse_another_masks_graph(self):
        first = np.zeros((96, 96), dtype=np.uint8)
        first[10, 10:51] = 255
        second = np.zeros_like(first)
        second[20:71, 20] = 255
        result1 = self._analyze(first)
        result2 = self._analyze(second)
        self.assertEqual(float(result1.summary_df.total_length.sum()), 40.0)
        self.assertEqual(float(result2.summary_df.total_length.sum()), 50.0)
        self.assertFalse(np.any(result2.skeleton & (second == 0)))
        repeated = self._analyze(first.copy())
        np.testing.assert_array_equal(repeated.skeleton, result1.skeleton)

    def test_cache_key_tracks_preprocessed_content_at_same_path(self):
        cfg = PhenotypingConfig(spur_max_length=0)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[5, 3:20] = 255
        _, first, first_key = pipeline_stages.skeletonize_union_mask("same-path.png", mask, cfg, None)
        mask[5] = 0
        mask[10, 3:20] = 255
        _, second, second_key = pipeline_stages.skeletonize_union_mask("same-path.png", mask, cfg, None)
        self.assertNotEqual(first_key, second_key)
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(np.all(second[10, 3:20]))

    def test_mutating_pipeline_result_does_not_poison_cached_analysis(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20, 10:50] = 255
        for cache_hit in (False, True):
            with self.subTest(cache_hit=cache_hit):
                clear_skeleton_cache()
                result = self._analyze(mask)
                if cache_hit:
                    result = self._analyze(mask.copy())
                result.skeleton[:] = False
                u, v = next(iter(result.G.edges))
                result.G[u][v]["length"] = 999
                result.G[u][v]["pixels"].clear()
                repeated = self._analyze(mask.copy())
                self.assertEqual(int(repeated.skeleton.sum()), 40)
                self.assertEqual(float(repeated.summary_df.total_length.sum()), 39.0)
                self.assertGreater(len(repeated.G[u][v]["pixels"]), 0)

    def test_mutating_skeleton_stage_input_or_output_does_not_poison_cache(self):
        cfg = PhenotypingConfig(spur_max_length=0, connect_broken_roots=False)
        original = np.zeros((24, 24), dtype=np.uint8)
        original[5, 3:20] = 255
        input_mask = original.copy()
        output_mask, output_skeleton, _ = pipeline_stages.skeletonize_union_mask(
            "same-path.png", input_mask, cfg, None
        )
        input_mask[:] = 0
        output_mask[:] = 0
        output_skeleton[:] = False
        for _ in range(2):
            restored_mask, restored_skeleton, _ = pipeline_stages.skeletonize_union_mask(
                "same-path.png", original.copy(), cfg, None
            )
            np.testing.assert_array_equal(restored_mask, original)
            np.testing.assert_array_equal(restored_skeleton, original > 0)
            restored_mask[:] = 0
            restored_skeleton[:] = False

    @staticmethod
    def _broken_mask():
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[20, 10:40] = 255
        mask[20, 45:75] = 255
        return mask

    def test_endpoint_search_points_outward_and_connects_gap(self):
        mask = self._broken_mask()
        output_mask, output_skeleton = connect_roots.connect_broken_roots(mask, mask > 0, max_search=20)
        self.assertTrue(np.all(output_mask[20, 10:75] == 255))
        self.assertEqual(int(output_skeleton.sum()), 65)
        self.assertEqual(int(np.count_nonzero(mask)), 60)

    def test_connection_respects_search_distance(self):
        mask = self._broken_mask()
        output_mask, _ = connect_roots.connect_broken_roots(mask, mask > 0, max_search=3)
        np.testing.assert_array_equal(output_mask, mask)

    def test_thick_horizontal_and_vertical_root_gaps_connect(self):
        for thickness in (3, 5, 7, 11):
            for vertical in (False, True):
                with self.subTest(thickness=thickness, vertical=vertical):
                    mask = np.zeros((140, 180), dtype=np.uint8)
                    mask[60:60 + thickness, 20:71] = 255
                    mask[60:60 + thickness, 76:127] = 255
                    if vertical:
                        mask = mask.T.copy()
                    skel = skeleton.skeletonize_mask(mask)
                    output_mask, output_skeleton = connect_roots.connect_broken_roots(
                        mask, skel, max_search=40
                    )
                    self.assertEqual(cv2.connectedComponents(output_skeleton.astype(np.uint8))[0] - 1, 1)
                    self.assertEqual(int(np.count_nonzero(output_mask) - np.count_nonzero(mask)), 5)

    def test_nearby_roots_must_face_each_other_and_the_bridge(self):
        for perpendicular in (False, True):
            with self.subTest(perpendicular=perpendicular):
                mask = np.zeros((140, 180), dtype=np.uint8)
                mask[60, 20:71] = 255
                if perpendicular:
                    mask[40:91, 76:83] = 255
                else:
                    # 射线命中粗根外缘，但其端点偏离连接方向超过角度阈值。
                    mask[60:91, 76:127] = 255
                skel = skeleton.skeletonize_mask(mask)
                hit, _, _ = connect_roots._ray_hit_foreground(mask, 60, 70, 0, 1, 40)
                self.assertTrue(hit)
                output_mask, _ = connect_roots.connect_broken_roots(mask, skel, max_search=40)
                np.testing.assert_array_equal(output_mask, mask)

    def test_thick_root_connection_still_respects_gap_and_search_limits(self):
        mask = np.zeros((140, 180), dtype=np.uint8)
        mask[60:71, 20:71] = 255
        mask[60:71, 76:127] = 255
        skel = skeleton.skeletonize_mask(mask)
        for limits in ({"max_search": 3}, {"max_gap": 5}, {"min_gap": 100}):
            with self.subTest(limits=limits):
                output_mask, _ = connect_roots.connect_broken_roots(mask, skel, **limits)
                np.testing.assert_array_equal(output_mask, mask)

    def test_batch_connects_before_largest_component_selection(self):
        mask = self._broken_mask()
        mask[80, 10:50] = 255  # 单个断片比它小，连接后的完整根比它大。
        result = self._analyze(mask, connect=True)
        self.assertTrue(np.all(result.skeleton[20, 10:75]))
        self.assertFalse(np.any(result.skeleton[80]))
        self.assertEqual(float(result.summary_df.total_length.sum()), 64.0)

    def test_seeded_connection_includes_unselected_fragment(self):
        mask = self._broken_mask()
        mask[80, 10:50] = 255
        result = self._analyze(mask, seeds=[(20, 15)], connect=True)
        self.assertTrue(np.all(result.skeleton[20, 10:75]))
        self.assertFalse(np.any(result.skeleton[80]))
        # 复用同一全图连接缓存时，图缓存仍必须区分不同种子选择。
        other = self._analyze(mask, seeds=[(80, 15)], connect=True)
        self.assertFalse(np.any(other.skeleton[20]))
        self.assertTrue(np.all(other.skeleton[80, 10:50]))

    def test_disabled_connection_keeps_selected_component_only(self):
        mask = self._broken_mask()
        result = self._analyze(mask, seeds=[(20, 15)], connect=False)
        self.assertEqual(int(result.skeleton.sum()), 30)
        self.assertFalse(np.any(result.skeleton[20, 40:]))

    def test_branch_angle_uses_local_pixels_and_is_insertion_order_independent(self):
        center, up, down, side = (5, 5), (0, 5), (10, 5), (8, 8)
        edges = [
            (center, up, [(5, 5), (4, 5), (0, 5)], 1),
            (center, down, [(10, 5), (6, 5), (5, 5)], 1),
            # 远端连线是 45°，连接处像素方向是 90°；像素存储顺序也允许反向。
            (center, side, [(8, 8), (5, 6), (5, 5)], 2),
        ]
        for ordered in itertools.permutations(edges):
            graph, levels = nx.Graph(), {}
            for u, v, pixels, level in ordered:
                graph.add_edge(u, v, pixels=pixels, length=10, diameter=2)
                levels[_edge_key(u, v)] = level
            table = stats.compute_level_stats(graph, levels).set_index("level")
            self.assertAlmostEqual(table.loc[2, "avg_branch_angle_deg"], 90.0)
            self.assertEqual(table.loc[1, "avg_branch_angle_deg"], 0)

    def test_zero_degree_branch_is_included_in_average(self):
        center, parent, parallel, perpendicular = (5, 5), (0, 5), (10, 6), (5, 10)
        graph, levels = nx.Graph(), {}
        for neighbor, pixels, level in (
            (parent, [(5, 5), (4, 5), parent], 1),
            (parallel, [(5, 5), (6, 5), parallel], 2),
            (perpendicular, [(5, 5), (5, 6), perpendicular], 2),
        ):
            graph.add_edge(center, neighbor, pixels=pixels, length=10, diameter=2)
            levels[_edge_key(center, neighbor)] = level
        table = stats.compute_level_stats(graph, levels).set_index("level")
        self.assertAlmostEqual(table.loc[2, "avg_branch_angle_deg"], 45.0)

    def test_secondary_continuation_nodes_do_not_add_branch_angles(self):
        center, parent, junction, tip = (5, 5), (0, 5), (5, 10), (10, 10)
        graph, levels = nx.Graph(), {}
        for u, v, level in ((parent, center, 1), (center, junction, 2), (junction, tip, 2)):
            graph.add_edge(u, v, pixels=[u, v], length=10, diameter=2)
            levels[_edge_key(u, v)] = level
        table = stats.compute_level_stats(graph, levels).set_index("level")
        self.assertAlmostEqual(table.loc[2, "avg_branch_angle_deg"], 90.0)


if __name__ == "__main__":
    unittest.main()
