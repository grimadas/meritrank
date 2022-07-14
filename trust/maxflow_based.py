"""
The code in this file is an implementation of the MaxFlow-Based Trust Functions.
used as a reputation mechanism in P2P networks.
"""
import igraph as ig
import networkx as nx
import numpy as np

from typing import Union


class MaxFlow:

    def __init__(self, graph: nx.DiGraph, seed_node: int = 0, normalized: bool = True) -> None:
        """Maxflow score trust function. Uses networkx's maxflow implementation
        https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.maximum_flow.html.
        @param graph: networkx directed graph
        @param alpha: weight of maxflow score
        """
        self.graph = graph

        # Precompute maxflow scores for all nodes if normalization is required
        if normalized:
            self.maxflow_scores = {}
            for t in graph.nodes():
                if t != seed_node:
                    self.maxflow_scores[t] = nx.maximum_flow_value(self.graph, seed_node, t, capacity='weight')
            m_sum = sum(self.maxflow_scores.values())
            m_sum = max(1, m_sum)
            for v in list(self.maxflow_scores.keys()):
                self.maxflow_scores[v] /= m_sum 
                
    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute maxflow score of target node from perspective of seed_node.
        @param seed_node: seed node id in a graph (int)
        @param target_node: target node id in a graph (int)
        @return: maxflow score of paths from seed_node to target_node
        """
        if seed_node == target_node:
            return 1.0
        if self.maxflow_scores:
            return self.maxflow_scores[target_node]
        maxflow_seed_target = nx.maximum_flow_value(self.graph, seed_node, target_node, capacity='weight')
        return maxflow_seed_target * self.alpha


class BarterCast:

    def __init__(self, graph: Union[nx.Graph, ig.Graph], use_igraph: bool = False) -> None:
        """Bartercast score trust function. https://ieeexplore.ieee.org/document/5160954
        @param graph: networkx graph or igraph graph
        @param use_igraph: use igraph implementation of maxflow (default: False).
        """
        self.graph = graph
        self.use_igraph = use_igraph

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute bartercast score of target node from perspective of seed_node.
        @param seed_node: seed node id in a graph (int)
        @param target_node: target node id in a graph (int)
        @return: bartercast score of paths from seed_node to target_node
        """
        if seed_node == target_node:
            return 1.0
        if self.use_igraph:
            maxflow_seed_target = self.graph.maxflow_value(seed_node, target_node, capacity='weight')
            maxflow_target_seed = self.graph.maxflow_value(target_node, seed_node, capacity='weight')
        else:
            maxflow_seed_target = nx.maximum_flow(self.graph, seed_node, target_node, capacity='weight')[0]
            maxflow_target_seed = nx.maximum_flow(self.graph, target_node, seed_node, capacity='weight')[0]
        values = float(np.arctan(maxflow_seed_target - maxflow_target_seed)) / float(0.5 * np.pi)
        return values


class RawBarterCast:

    def __init__(self, graph: nx.DiGraph, alpha: float = 1.0) -> None:
        """A modification of Bartercast score trust function without arctan.
        Score = maxflow_seed_target - maxflow_target_seed.
        @param graph: networkx directed graph
        @param alpha: weight of bartercast score
        """
        self.alpha = alpha
        self.graph = graph

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute bartercast score of target node from perspective of seed_node
        @param seed_node: seed node id in a graph (int)
        @param target_node: target node id in a graph (int)
        @return: bartercast score of paths from seed_node to target_node
        """
        if seed_node == target_node:
            return 1.0
        maxflow_seed_target = nx.maximum_flow(self.graph, seed_node, target_node, capacity='weight')[0]
        maxflow_target_seed = nx.maximum_flow(self.graph, target_node, seed_node, capacity='weight')[0]
        values = float(maxflow_seed_target - maxflow_target_seed)
        return values


class BoundedBarterCast:

    def __init__(self, graph: Union[nx.DiGraph, ig.Graph], 
                    alpha: float = 1.0, 
                    use_igraph: bool = False, 
                    bound: float = 1000) -> None:
        """A modification of Bartercast score trust weighted by the net contribution of the target node and seed node.
        @param graph: networkx directed graph or igraph graph
        @param alpha: weight of bartercast score
        @param use_igraph: use igraph implementation of maxflow (default: False).
        @param bound: bound of net contribution of a node (default: 1000)
        """
        self.alpha = alpha
        self.graph = graph

        self.use_igraph = use_igraph

        self.scores = {}

    def net_contrib(self, node: int) -> float:
        """Compute net contribution of a node.
        score = min(alpha * out_degree(node) + 1 - in_degree(node), 1000)
        @param node: node id in a graph (int)
        @return: net contribution of a node
        """
        if self.use_igraph:
            out_deg = self.graph.strength(node, mode='OUT', weights='weight')
            in_deg = self.graph.strength(node, mode='IN', weights='weight')
        else:
            out_deg = self.graph.out_degree(node, weight='weight')
            in_deg = self.graph.in_degree(node, weight='weight')

        return min(self.alpha * (out_deg + 1) - in_deg, 1000)

    def _calc(self, seed_node: int, target_node: int) -> None:
        if self.use_igraph:
            val = self.graph.maxflow_value(seed_node, target_node, capacity='weight')
        else:
            val = nx.maximum_flow_value(self.graph, seed_node, target_node, capacity='weight')
        self.scores[seed_node][target_node] = val

    def calc(self, seed_node: int, target_node: int) -> float:
        if seed_node not in self.scores:
            self.scores[seed_node] = {}
            self._calc(seed_node, target_node)
        if target_node not in self.scores[seed_node]:
            self._calc(seed_node, target_node)
        return self.scores[seed_node][target_node]

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute score of target node from perspective of seed_node.
        @param seed_node: seed node id in a graph (int)
        @param target_node: target node id in a graph (int)
        @return: trust score of target node from perspective of seed_node"""
        if seed_node == target_node:
            return 1.0
        p1 = 0.0
        coef = self.net_contrib(seed_node)
        if coef > 0:
            p1 = coef * self.calc(seed_node, target_node)

        p2 = 0.0
        coef = self.net_contrib(target_node)
        if coef > 0:
            p2 = coef * self.calc(target_node, seed_node)
        values = float(p1 - p2)
        return values


class PenaltyCast:

    def __init__(self, graph: nx.DiGraph, alpha: float = 2.0) -> None:
        """A modification of Bartercast score trust function with penalty.
        Score = maxflow_seed_target - maxflow_target_seed.
        Two maxflows are computed: one from seed_node to target_node and one from target_node to seed_node.
        The penalty is computed as the difference between the two maxflows.
        @param graph: networkx directed graph
        @param alpha: weight of bartercast score
        """
        self.graph = graph

        self.scores = {}
        self.path_counts = {}

        self.aux_scores = {}

        self.penalites = {}
        self.alpha = alpha

        self.auxes = {}

    def _calc(self, seed_node: int, target_node: int) -> None:
        val = nx.maximum_flow(self.graph, seed_node, target_node, capacity='weight')
        self.scores[seed_node][target_node] = val[0]

        raw_count = {k: sum(v.values()) for k, v in val[1].items()}
        total_sum = sum(raw_count.values()) - raw_count[seed_node]
        norm_count = {k: v / total_sum for k, v in raw_count.items() if v > 0 and k != seed_node}
        self.path_counts[seed_node][target_node] = norm_count

    def calc(self, seed_node: int, target_node: int) -> float:
        if seed_node not in self.scores:
            self.scores[seed_node] = {}
            self.path_counts[seed_node] = {}
            self._calc(seed_node, target_node)
        if target_node not in self.scores[seed_node]:
            self._calc(seed_node, target_node)
        return self.scores[seed_node][target_node]

    # build aux graph
    def aux_graph(self, seed_node: int) -> float:
        if seed_node in self.auxes:
            return self.auxes[seed_node]

        penalties = {}
        for k in self.graph.pred[seed_node]:
            self.calc(seed_node, target_node=k)
            w = self.graph[k][seed_node]['weight']

            for i, v in self.path_counts[seed_node][k].items():
                if i not in penalties:
                    penalties[i] = 0
                penalties[i] += v * w / self.alpha

        self.auxes[seed_node] = self.graph.copy()
        for k, v in penalties.items():
            if k in self.auxes[seed_node][seed_node]:
                self.auxes[seed_node][seed_node][k]['weight'] -= v
        return self.auxes[seed_node]

    def _aux_calc(self, seed_node: int, target_node: int) -> None:
        val = nx.maximum_flow_value(self.auxes[seed_node], seed_node, target_node, capacity='weight')
        self.aux_scores[seed_node][target_node] = val

    def recalc_penalites(self, seed_node: int, neigh_node: int) -> float:
        self._calc(seed_node, target_node=neigh_node)
        w = self.graph[neigh_node][seed_node]['weight']

        self.aux_graph(seed_node)
        for i, v in self.path_counts[seed_node][neigh_node].items():
            if i in self.auxes[seed_node][seed_node]:
                self.auxes[seed_node][seed_node][i]['weight'] -= v * w / self.alpha

    def aux_calc(self, seed_node: int, target_node: int) -> float:
        self.aux_graph(seed_node)
        if seed_node not in self.aux_scores:
            self.aux_scores[seed_node] = {}
            self._aux_calc(seed_node, target_node)
        if target_node not in self.aux_scores[seed_node]:
            self._aux_calc(seed_node, target_node)
        return self.aux_scores[seed_node][target_node]

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute score of target node from perspective of seed_node
        @param seed_node: seed node id in a graph (int)
        @param target_node: target node id in a graph (int)
        @return: score of target node from perspective of seed_node
        """
        if seed_node == target_node:
            return 1.0
        maxflow_target_seed = self.calc(target_node, seed_node)
        maxflow_seed_target = self.aux_calc(seed_node, target_node)
        values = maxflow_seed_target - maxflow_target_seed
        return values

class Netflow:

    def __init__(self, graph: nx.Graph, seed_node: int = None, alpha: float = 2) -> None:
        """
        This class implements the Netflow algorithm. As described in Trustchain paper. 
        """
        self.graph = graph
        self.alpha = alpha
        self.seed_node = seed_node

        self._compute_scores()

    def _prepare(self) -> None:
        self._graph = self.graph.copy()

        for neighbour in self._graph.out_edges([self.seed_node], 'weight', 0):
            cap = self._graph.adj[self.seed_node][neighbour[1]]['weight']
            self._graph.adj[self.seed_node][neighbour[1]]['weight'] = float(cap) / float(self.alpha)

    def _initial_step(self) -> None:
        """
        In the intial step, all capactities are computed
        """

        for node in self._graph.nodes():
            self._compute_capacity(node)
        return self._graph

    def _compute_capacity(self, node: int) -> None:
        if node == self.seed_node:
            return
        contribution = nx.maximum_flow_value(self._graph, self.seed_node, node, 'weight')
        consumption = nx.maximum_flow_value(self._graph, node, self.seed_node, 'weight')

        self._graph.add_node(node, weight=max(0, contribution - consumption))
        self._graph.add_node(node, bartercast=contribution - consumption)

    def _netflow_step(self):

        compute_score = lambda node: nx.maximum_flow_value(self._graph, self.seed_node, node,
                                                           'weight') if node != self.seed_node else 0

        scores = {node: compute_score(node) for node in self._graph.nodes()}
        nx.set_node_attributes(self._graph, scores, 'score')

    def _compute_scores(self):
        self._prepare()
        self._initial_step()
        self._netflow_step()

    def compute(self, seed_node: int, target_node: int) -> float:
        if seed_node != self.seed_node:
            self.seed_node = seed_node
            self._compute_scores()
        scores = nx.get_node_attributes(self._graph, 'score')
        return scores[target_node]
