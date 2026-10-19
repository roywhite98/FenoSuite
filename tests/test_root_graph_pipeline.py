# -*- coding: utf-8 -*-
"""根系图：ridge / 曲率关键点 / 简化 / decross 基础测试。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import unittest

from plant_phenotype.root_phenotype.core import ridge as ridge_mod
from plant_phenotype.root_phenotype.core import topology
from plant_phenotype.root_phenotype.core import graph
from plant_phenotype.root_phenotype.core import simplify_graph
from plant_phenotype.root_phenotype.core import skeleton_prune


class TestRootGraphPipeline(unittest.TestCase):
    def test_ridge_integrate_constant(self):
        R = np.ones((10, 10), dtype=np.float32)
        path = [(2, 2), (2, 3), (2, 4)]
        integral, mean = ridge_mod.integrate_ridge_along_polyline(R, path)
        self.assertAlmostEqual(mean, 1.0, places=5)
        self.assertGreater(integral, 0)

    def test_curvature_topology_runs(self):
        skel = np.zeros((5, 8), dtype=bool)
        for c in range(5):
            skel[2, c] = True
        for r in range(2, 5):
            skel[r, 4] = True
        _ep, junc, seg, _eid, _jid = topology.build_topology(
            skel, junction_min_neighbors=3, curvature_angle_deg=45.0
        )
        self.assertGreaterEqual(len(seg), 1)
        self.assertIsInstance(junc, list)

    def test_simplify_removes_short_leaf(self):
        import networkx as nx

        G = nx.Graph()
        G.add_node((0, 0), pos=(0, 0))
        G.add_node((0, 1), pos=(0, 1))
        G.add_edge(
            (0, 0),
            (0, 1),
            pixels=[(0, 0), (0, 1)],
            length=1.0,
            ridge_integral=0.01,
            weight=1.0,
        )
        H = simplify_graph.prune_leaf_edges_by_score(G, min_score=0.5, score_attr="ridge_integral")
        self.assertEqual(H.number_of_nodes(), 1)

    def test_build_weighted_graph_ridge_field_no_crash(self):
        skel = np.zeros((4, 6), dtype=bool)
        skel[1, 1:5] = True
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[skel] = 255
        R = ridge_mod.compute_multiscale_r_max(mask, sigmas=(1.0, 2.0))
        G, _ep, _ju, _eid, _jid = graph.build_weighted_graph(
            skel,
            root_mask=mask,
            ridge_field=R,
            curvature_angle_deg=None,
            decross_enabled=False,
            simplify_enabled=False,
        )
        self.assertGreaterEqual(G.number_of_edges(), 1)
        _, _, data = next(iter(G.edges(data=True)))
        self.assertIn("ridge_integral", data)

    def test_spur_prune_removes_one_pixel_branch(self):
        # 三条互不相邻的射线；顶端毛刺只含一个像素，下面两臂更长。
        skel = np.zeros((7, 7), dtype=bool)
        skel[2, 3] = skel[3, 3] = True
        skel[4, 2] = skel[5, 1] = True
        skel[4, 4] = skel[5, 5] = True
        out = skeleton_prune.prune_skeleton_terminal_spurs(skel, max_spur_length=1, junction_min_neighbors=3)
        self.assertFalse(out[2, 3])
        self.assertTrue(out[3, 3])
        self.assertTrue(out[5, 1])
        self.assertTrue(out[5, 5])

    def test_spur_prune_preserves_isolated_line(self):
        skel = np.zeros((5, 6), dtype=bool)
        skel[2, 0:4] = True
        out = skeleton_prune.prune_skeleton_terminal_spurs(skel, max_spur_length=8)
        np.testing.assert_array_equal(out, skel)

    def test_merge_clustered_endpoints_three_to_one(self):
        import networkx as nx

        J = (10, 10)
        leaves = [(10, 11), (10, 12), (11, 11)]
        G = nx.Graph()
        G.add_node(J, pos=J)
        for L in leaves:
            G.add_node(L, pos=L)
            G.add_edge(
                L,
                J,
                pixels=[L, J],
                length=2.0,
                weight=1.0,
                ridge_integral=2.0,
                diameter=1.0,
                curvature=0.0,
            )
        H = simplify_graph.merge_clustered_endpoints(G, max_dist_px=4.0)
        # 合并后中心 J 自身也成为度为 1 的节点；只统计原叶端的代表。
        tips = [n for n in H.nodes() if H.degree(n) == 1 and n != J]
        self.assertEqual(len(tips), 1)
        self.assertEqual(H.number_of_edges(), 1)
        self.assertEqual(H.degree(J), 1)

    def test_endpoint_merge_off_preserves_multi_leaf_star(self):
        import networkx as nx

        J = (0, 0)
        leaves = [(0, 1), (0, 2), (1, 1)]
        G = nx.Graph()
        G.add_node(J, pos=J)
        for L in leaves:
            G.add_node(L, pos=L)
            G.add_edge(L, J, pixels=[L, J], length=1.0, weight=1.0)
        H = simplify_graph.merge_clustered_endpoints(G, max_dist_px=0.0)
        self.assertEqual(H.number_of_nodes(), G.number_of_nodes())

    def test_find_keypoints_vectorized_matches_bruteforce(self):
        skel = np.zeros((7, 9), dtype=bool)
        skel[3, 1:7] = True
        skel[1:6, 4] = True
        jmin = 3
        ep_v, ju_v, km_v = topology.find_keypoints(skel, junction_min_neighbors=jmin)

        def _bruteforce(s, junction_min_neighbors=3):
            endpoints, junctions = [], []
            keypoint_mask = np.zeros_like(s, dtype=bool)
            h, w = s.shape
            for r in range(h):
                for c in range(w):
                    if not s[r, c]:
                        continue
                    nc = topology._neighbor_count(s, r, c)
                    if nc == 1:
                        endpoints.append((r, c))
                        keypoint_mask[r, c] = True
                    elif nc >= junction_min_neighbors:
                        junctions.append((r, c))
                        keypoint_mask[r, c] = True
            return endpoints, junctions, keypoint_mask
        ep_b, ju_b, km_b = _bruteforce(skel, jmin)
        self.assertEqual(set(ep_v), set(ep_b))
        self.assertEqual(set(ju_v), set(ju_b))
        np.testing.assert_array_equal(km_v, km_b)

    def test_ensure_simple_path_on_graph_removes_walk_loop(self):
        import networkx as nx
        from plant_phenotype.root_phenotype.core import main_root

        G = nx.Graph()
        for a, b in [((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 1), (0, 3))]:
            G.add_edge(a, b, weight=1.0, length=1.0)
        walk = [(0, 0), (0, 1), (0, 2), (0, 1), (0, 3)]
        out = main_root.ensure_simple_path_on_graph(G, walk)
        self.assertEqual(out, [(0, 0), (0, 1), (0, 3)])


if __name__ == "__main__":
    unittest.main()
