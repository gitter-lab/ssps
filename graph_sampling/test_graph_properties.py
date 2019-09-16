import unittest
import numpy as np
import graph_sampling.graph_properties as gp


class GraphPropertiesTests(unittest.TestCase):

    def setUp(self) -> None:

        self.linear_ugraph = np.array([[0,1,0,0],
                                  [0,0,1,0],
                                  [0,0,0,1],
                                  [0,0,0,0]])

        self.linear_dgraph = np.array([[0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,0],
                                       [0,0,1,0]])

        self.cyclic_ugraph = np.array([[0, 1, 0, 0, 0],
                                       [0, 0, 1, 0, 0],
                                       [0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0]])

        self.cyclic_dgraph = np.array([[0, 1, 0, 0, 0],
                                       [0, 0, 1, 0, 0],
                                       [0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0],
                                       [1, 0, 0, 1, 0]])

        self.unconnected_graph = np.array([[0,1,0,0],
                                           [0,0,0,0],
                                           [0,0,0,1],
                                           [0,0,0,0]])

        self.tree = np.array([[0,1,1,0,0],
                              [0,0,0,1,1],
                              [0,0,0,0,0],
                              [0,0,0,0,0],
                              [0,0,0,0,0]])

        self.triangle = np.array([[0,1,0],
                                  [0,0,1],
                                  [1,0,0]])

    def test_dfs(self):

        self.assertEqual(gp.dfs(self.linear_ugraph),
                         [0,1,2,3])
        self.assertEqual(gp.dfs(self.linear_dgraph, 0, directed=True),
                                [0,1,2])
        self.assertEqual(gp.dfs(self.linear_dgraph, 0, directed=False),
                                [0,1,2,3])
        self.assertEqual(gp.dfs(self.linear_dgraph, 3, directed=True),
                                [3, 2])
        self.assertEqual(gp.dfs(self.tree, 0, directed=True),
                         [0,2,1,4,3])

    def test_is_connected(self):

        self.assertTrue(gp.is_connected(self.linear_ugraph))
        self.assertTrue(gp.is_connected(self.linear_dgraph))
        self.assertTrue(not gp.is_connected(self.unconnected_graph))

    def test_all_reachable(self):

        self.assertTrue(gp.all_reachable(self.linear_ugraph, 0))
        self.assertTrue(not gp.all_reachable(self.linear_dgraph, 0))
        self.assertTrue(gp.all_reachable(self.cyclic_ugraph, 0))
        self.assertTrue(not gp.all_reachable(self.cyclic_dgraph, 0))
        self.assertTrue(gp.all_reachable(self.cyclic_dgraph, 4))
        self.assertTrue(not gp.all_reachable(self.unconnected_graph, 0))

    def test_is_dag(self):

        self.assertTrue(gp.is_dag(self.cyclic_dgraph))
        self.assertTrue(not gp.is_dag(self.cyclic_ugraph))
        self.assertTrue(gp.is_dag(self.tree))
        self.assertTrue(not gp.is_dag(self.triangle))

if __name__ == '__main__':
    unittest.main()
