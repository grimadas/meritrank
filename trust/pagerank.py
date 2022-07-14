from collections import defaultdict
from math import ceil

import networkx as nx

from .random_walks import RandomWalks, BiasStrategies


class PersonalizedPageRankNx:
    """
    This class implements the personalized pagerank
    """

    def __init__(self, graph: nx.Graph, seed_node: int = 0, seed_weight: float = 1.0, **kwargs) -> None:
        self.graph = graph
        self.seed_node = seed_node
        self.seed_weight = seed_weight

        self.params = kwargs
        self._recompute_pagerank()

    def _recompute_pagerank(self) -> float:
        self.rank = nx.pagerank(self.graph, personalization={self.seed_node: self.seed_weight}, **self.params)

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute personal pagerank from seed_node to target_node"""
        if self.seed_node != seed_node:
            self.seed_node = seed_node
            self._recompute_pagerank()
        return self.rank[target_node]

class PersonalizedPageRank:

    def __init__(self, graph: nx.Graph, seed_node: int = None, 
                 number_random_walks: int = 1000,
                 reset_probability: float = 0.15) -> None:
        """This class implements a Monte Carlo implementation of the hitting time algorithm
        by running random walks in a networkx graph.
        @param graph: The networkx graph to run the random walks on.
        @param seed_node: The node to start the random walks from.
        @param number_random_walks: The number of random walks to run.
        @param reset_probability: The probability of resetting the random walk.
        """
        self.graph = graph
        self.number_random_walks = number_random_walks
        self.reset_probability = reset_probability
        self.seed_node = seed_node

        self.random_walks = RandomWalks(self.graph)
        self.random_walks.run(seed_node, int(self.number_random_walks), self.reset_probability)

    def compute(self, seed_node: int, target_node: int) -> float:
        if not self.random_walks.has_node(seed_node):
            self.random_walks.run(seed_node,
                                  int(self.number_random_walks),
                                  self.reset_probability)

        total_hits = self.random_walks.get_total_hits(seed_node, target_node) 
        all_hits = self.random_walks.get_total_sum(seed_node) - self.random_walks.get_total_hits(seed_node, seed_node) 
        all_hits = max(1, all_hits)
        return total_hits / all_hits


class BoundedPPR(PersonalizedPageRank):

    def calculate_penalty(self, s, t, value):
        encounters = defaultdict(int)
        penalties = defaultdict(int)
        for k in self.random_walks.random_walks[s]:
            indexes = [i for i, x in enumerate(k) if x == t]
            if indexes:
                start_val = 1
                for v in indexes:
                    for p in range(start_val, v + 1):
                        encounters[k[p]] += 1
        # calculate fractions
        for k, v in encounters.items():
            penalties[k] += ceil(value * v / encounters[t])
        return penalties


    def compute(self, seed_node: int, target_node: int) -> float:
        if not self.random_walks.has_node(seed_node):
            self.random_walks.run(seed_node,
                                  int(self.number_random_walks),
                                  self.reset_probability)

        # Calculate the penalities

        total_hits = self.random_walks.get_total_hits(seed_node, target_node) 
        all_hits = self.random_walks.get_total_sum(seed_node) - self.random_walks.get_total_hits(seed_node, seed_node) 
        all_hits = max(1, all_hits)
        return total_hits / all_hits




class TrustRank:

    def __init__(self, graph: nx.Graph,
                 number_random_walks: int = 10000,
                 reset_probability: float = 0.33,
                 alpha: float = 1.0,
                 use_bias: bool = False,
                 update_weight: bool = False,
                 ) -> None:
        """
        This class implements the personalized pagerank scaled by the net contribution of the seed node and deducts
        the personalized PageRank of the source node from the seed node, again scaled by its net contribution.
        @param graph: The networkx graph to run the random walks on.
        @param number_random_walks: The number of random walks to run.
        @param reset_probability: The probability of resetting the random walk.
        @param alpha: The alpha parameter for the biased random walks.
        @param use_bias: Whether to use biased random walks. Look ALPHA_DIFF strategy in RandomWalks for more details (default: false).
        @param update_weight: Keep track of the weight of the edges (default: False).
        """
        self.graph = graph
        self.number_random_walks = number_random_walks
        self.reset_probability = reset_probability
        self.alpha = alpha
        self.use_bias = use_bias
        self.update_weight = update_weight

        self.random_walks = RandomWalks(self.graph, alpha)
        self.reverse_walks = RandomWalks(self.graph, alpha)

    def compute(self, seed_node: int, target_node: int) -> float:

        if not self.random_walks.has_node(seed_node):
            bias_strategy = BiasStrategies.ALPHA_DIFF if self.use_bias else BiasStrategies.EDGE_WEIGHT
            self.random_walks.run(seed_node, self.number_random_walks,
                                  self.reset_probability, bias_strategy=bias_strategy, update_weight=self.update_weight)
            self.reverse_walks.run(seed_node, self.number_random_walks, self.reset_probability,
                                   back_random_walk=True,  bias_strategy=bias_strategy, update_weight=self.update_weight)

        # Process random walks with weighted PHT: number of hits of a target node
        pr1 = self.random_walks.get_number_of_hits(seed_node,
                                                   target_node) / self.number_random_walks
        pr2 = self.reverse_walks.get_number_of_hits(seed_node, target_node) / self.number_random_walks
        return pr1 - pr2


class ReciprocalScaledPageRank:
    
    def __init__(self, graph: nx.DiGraph,
                 base_number_of_walks: int = 10,
                 reset_probability: float = 0.1,
                 alpha: float = 2.0
                 ) -> None:
        """
        This class implements the personalized pagerank scaled by the net contribution of the seed node and deducts
        the personalized PageRank of the source node from the seed node, again scaled by its net contribution.
        """
        self.graph = graph
        self.number_random_walks = base_number_of_walks
        self.reset_probability = reset_probability
        self.alpha = alpha

        self.random_walks = RandomWalks(self.graph, alpha, base_number_of_walks)

    def net_contrib(self, node: int) -> float:
        return min(self.graph.out_degree(node, weight='weight'), 1000)

    def compute(self, seed_node: int, target_node: int) -> float:
        if not self.random_walks.has_node(seed_node):
            self.random_walks.run(seed_node,
                                  int(self.number_random_walks * self.net_contrib(seed_node)),
                                  self.reset_probability)

        if not self.random_walks.has_node(target_node):
            self.random_walks.run(target_node,
                                  int(self.number_random_walks * self.net_contrib(target_node)),
                                  self.reset_probability)

        pr1 = self.random_walks.get_total_hits(seed_node, target_node)
        pr2 = self.random_walks.get_total_hits(target_node, seed_node)
        return pr1 / self.number_random_walks - pr2 / self.number_random_walks


class WBPPageRank(ReciprocalScaledPageRank):

    def net_contrib(self, node: int) -> float:
        return min(self.alpha * (self.graph.out_degree(node, weight='weight') / 1000 + 1)
                   - self.graph.in_degree(node, weight='weight') / 1000, 1000)


class SBPPageRank:

    def __init__(self, graph: nx.DiGraph,
                 base_number_random_walks: int = 10,
                 reset_probability: float = 0.1,
                 alpha: float = 1.0,
                 self_manage_penalties: bool = False) -> None:
        """
        Implementation of the Personalised PageRank algorithm, whereby we scaled the random
        walks by the net contribution of the seed node. Additionally we compute the scaled personalised PagRanks of
        the seed node starting at the target node. The reputation scores produced by this reputation mechanism are given
        by the difference of the two. """

        self.graph = graph
        self.number_random_walks = base_number_random_walks
        self.reset_probability = reset_probability

        self.penalties = {}
        self.self_manage_penalties = self_manage_penalties
        self.alpha = alpha

        self.random_walks = RandomWalks(self.graph, alpha, self.number_random_walks)

    def net_contrib(self, node: int) -> float:
        return min(self.alpha * (self.graph.out_degree(node, weight='weight') + 1), 1000)

    def calculate_penalty(self, s, t, value):
        encounters = defaultdict(int)
        penalties = defaultdict(int)
        for k in self.random_walks.random_walks[s]:
            indexes = [i for i, x in enumerate(k) if x == t]
            if indexes:
                start_val = 1
                for v in indexes:
                    for p in range(start_val, v + 1):
                        encounters[k[p]] += 1
        # calculate fractions
        for k, v in encounters.items():
            penalties[k] += ceil(value * v / encounters[t])
        return penalties

    def add_penalties(self, seed_node, penalties):
        self.penalties[seed_node] = penalties

    def get_penalties(self, seed_node):
        if seed_node not in self.penalties:
            self.penalties[seed_node] = defaultdict(int)
        for e, _, w in self.graph.in_edges(seed_node, data=True):
            pen = self.calculate_penalty(seed_node, e, w['weight'])
            for k, v in pen.items():
                self.penalties[seed_node][k] += v
        return self.penalties[seed_node]

    def compute(self, seed_node: int, target_node: int) -> float:
        if not self.random_walks.has_node(seed_node):
            self.random_walks.run(seed_node,
                                  int(self.number_random_walks * self.net_contrib(seed_node)),
                                  self.reset_probability,
                                  bias_strategy=BiasStrategies.EDGE_WEIGHT_BOUNDED,
                                  penalties=self.penalties.get(seed_node, None),
                                  update_weight=False
                                  )

            if self.self_manage_penalties and seed_node not in self.penalties:
                self.get_penalties(seed_node)
                self.random_walks.run(seed_node,
                                      int(self.number_random_walks * self.net_contrib(seed_node)),
                                      self.reset_probability,
                                      bias_strategy=BiasStrategies.EDGE_WEIGHT_BOUNDED,
                                      penalties=self.penalties.get(seed_node, None),
                                      update_weight=False
                                      )

        if not self.random_walks.has_node(target_node):
            self.random_walks.run(target_node,
                                  int(self.number_random_walks * self.net_contrib(target_node)),
                                  self.reset_probability,
                                  bias_strategy=BiasStrategies.EDGE_WEIGHT_BOUNDED,
                                  penalties=None,
                                  update_weight=False
                                  )

        penalty = 0.0 if not self.penalties.get(seed_node) else self.penalties[seed_node].get(target_node, 0)
        pr1 = self.random_walks.get_total_hits(seed_node, target_node) - penalty / self.alpha
        pr2 = self.random_walks.get_total_hits(target_node, seed_node)
        return pr1 - pr2
