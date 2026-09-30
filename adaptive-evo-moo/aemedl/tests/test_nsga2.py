import unittest

import numpy as np

from adaptive_evo.nsga2 import (
    binary_tournament, crowding_distance, environmental_selection, non_dominated_sort,
)


class TestNSGA2(unittest.TestCase):
    def setUp(self) -> None:
        self.F = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 1.0], [3.0, 4.0], [5.0, 5.0]])

    def test_fronts_are_correct(self) -> None:
        fronts = non_dominated_sort(self.F)
        self.assertEqual(sorted(fronts[0]), [0, 1, 2])
        self.assertEqual(fronts[1], [3])
        self.assertEqual(fronts[2], [4])

    def test_front_zero_is_mutually_non_dominated(self) -> None:
        rng = np.random.default_rng(0)
        F = rng.random((40, 4))
        front = non_dominated_sort(F)[0]
        for i in front:
            for j in front:
                if i != j:
                    self.assertFalse(np.all(F[i] <= F[j]) and np.any(F[i] < F[j]))

    def test_every_point_in_exactly_one_front(self) -> None:
        F = np.random.default_rng(1).random((30, 3))
        flat = [i for f in non_dominated_sort(F) for i in f]
        self.assertEqual(sorted(flat), list(range(30)))

    def test_crowding_boundaries_infinite(self) -> None:
        d = crowding_distance(self.F, [0, 1, 2])
        self.assertTrue(np.isinf(d[0]) and np.isinf(d[2]))
        self.assertTrue(np.isfinite(d[1]) and d[1] > 0)

    def test_selection_size_and_elitism(self) -> None:
        idx, rank, _ = environmental_selection(self.F, 3)
        self.assertEqual(len(idx), 3)
        self.assertTrue(np.all(rank == 0))
        self.assertEqual(sorted(idx.tolist()), [0, 1, 2])

    def test_empty_input(self) -> None:
        self.assertEqual(non_dominated_sort(np.zeros((0, 2))), [])

    def test_tournament_prefers_lower_rank(self) -> None:
        rank = np.array([0, 5])
        crowd = np.array([0.0, 100.0])
        picks = binary_tournament(rank, crowd, np.random.default_rng(0), 500)
        self.assertGreater(np.mean(picks == 0), 0.6)


if __name__ == "__main__":
    unittest.main()
