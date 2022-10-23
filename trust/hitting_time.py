"""
The code in this file is an implementation of the Personalised Hitting Time Algorithm
used as a reputation mechanism in P2P networks.
Based on the ideas described in: 
https://dash.harvard.edu/bitstream/handle/1/33009675/liu_aamas16.pdf?sequence=1
"""

import networkx as nx

from .random_walks import RandomWalks


class PersonalizedHittingTime:

    def __init__(self, graph: nx.Graph, seed_node: int = None, number_random_walks: int = 1000,
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
        n_hits = self.random_walks.get_number_positive_hits(seed_node, target_node)
        n_sum = self.random_walks.get_number_positive_hits_sum(seed_node) - self.random_walks.get_number_positive_hits(
            seed_node, seed_node)
        n_sum = max(1, n_sum)
        return n_hits / n_sum


class BL_PHT:

    def __init__(self, graph: nx.Graph, seed_node: int = None,
                 number_of_positive_random_walks: int = 1000,
                 p_reset_probability: float = 0.15,
                 number_of_negative_random_walks: int = 500,
                 n_reset_probability: float = 0.3
                 ) -> None:
        """This class implements a Monte Carlo implementation of the hitting time algorithm
        by running random walks in a networkx graph.
        @param graph: The networkx graph to run the random walks on.
        @param seed_node: The node to start the random walks from.
        @param number_random_walks: The number of random walks to run.
        @param reset_probability: The probability of resetting the random walk.
        """
        self.graph = graph
        self.pnrw = number_of_positive_random_walks
        self.prp = p_reset_probability
        self.nnrw = number_of_negative_random_walks
        self.nrp = n_reset_probability

        self.seed_node = seed_node

        self.neg_repu_scores = {}
        self.random_walks = RandomWalks(self.graph)
        self.random_walks.seed_node = seed_node
        self.random_walks.run_with_all_negative_walks(seed_node, self.pnrw, self.prp,
                                                      self.nnrw, self.nrp)
        self.calc_negative_reputation_scores(seed_node)

    def calc_negative_reputation_scores(self, seed_node):
        neg_sum = self.random_walks.get_number_negative_hits_sum(seed_node)
        all_hits = max(1, self.random_walks.get_number_positive_hits_sum(seed_node)
                       - self.random_walks.get_number_positive_hits(seed_node, seed_node)
                       + neg_sum
                       )
        for k, nw in self.random_walks.neg_hits[seed_node].items():
            w = nw / all_hits
            k_hits = self.random_walks.get_number_positive_hits_sum(k)
            for p_i, p_w in self.random_walks.hits[k].items():
                self.neg_repu_scores[p_i] = w*p_w / k_hits

    def compute(self, seed_node: int, target_node: int) -> float:
        if self.random_walks.seed_node != seed_node:
            self.random_walks.run_with_all_negative_walks(seed_node, self.pnrw, self.prp,
                                                          self.nnrw, self.nrp)
            self.calc_negative_reputation_scores(seed_node)

        neg_sum = self.random_walks.get_number_negative_hits_sum(seed_node)
        all_hits = max(1, self.random_walks.get_number_positive_hits_sum(seed_node)
                       - self.random_walks.get_number_positive_hits(seed_node, seed_node)
                       + neg_sum
                       )

        total_hits = self.random_walks.get_number_positive_hits(seed_node, target_node)
        return total_hits / all_hits - self.neg_repu_scores.get(target_node, 0)
