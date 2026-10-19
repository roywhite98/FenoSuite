# -*- coding: utf-8 -*-
"""网络特征（来自 rootmask）。"""
import logging
import networkx as nx
from typing import Dict, Any

logger = logging.getLogger(__name__)


def compute_graph_features(G: nx.Graph, *, compute_betweenness: bool = False) -> None:
    if G.number_of_nodes() == 0:
        return
    deg = dict(G.degree())
    nx.set_node_attributes(G, deg, "degree")
    if compute_betweenness:
        n_nodes = len(G)
        if n_nodes <= 2000:
            logger.info("使用精确 betweenness centrality 计算，节点数：%d", n_nodes)
            betweenness = nx.betweenness_centrality(G, weight="weight")
        else:
            k = min(200, n_nodes)
            logger.info("使用近似 betweenness centrality 计算（k=%d），节点数：%d", k, n_nodes)
            betweenness = nx.betweenness_centrality(G, k=k, weight="weight")
        nx.set_node_attributes(G, betweenness, "betweenness")
    for n in G.nodes():
        wd = sum(G[n][nb].get("weight", 1.0) for nb in G.neighbors(n))
        G.nodes[n]["weighted_degree"] = wd
