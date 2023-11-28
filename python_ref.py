from typing import Callable, Sequence
import random
import heapq

class Heap[T]:
    """Heap wrapper cause I dislike the python heap API"""
    def __init__(self, data: Sequence[T], key: Callable[[T], float]):
        self.key = key
        self.heap = [(key(entry), entry) for entry in data]
        heapq.heapify(self.heap)

    def push(self, entry: T):
        heapq.heappush((self.key(entry), entry))

    def peak(self) -> T:
        return self.heap[0][1]

    def pop(self) -> T:
        return heapq.heappop(self.heap)[1]

    def __len__(self):
        return len(self.heap)

class Laesa[T]:
    """Computes approximate Nearest Neighbor with linear preprocessing.

    References
    ----------
    .. [1] M. L. Mico, J. Oncina, and E. Vidal,
        “A new version of the nearest-neighbour approximating and eliminating search algorithm (AESA) with linear preprocessing time and memory requirements,”
        Pattern Recognition Letters, vol. 15, no. 1, pp. 9-17, Jan. 1994, doi: 10.1016/0167-8655(94)90095-7.
    .. [2] F. Moreno-Seco, L. Mico, and J. Oncina,
        “A modification of the LAESA algorithm for approximated k-NN classification,”
        Pattern Recognition Letters, vol. 24, no. 1, pp. 47-53, Jan. 2003, doi: 10.1016/S0167-8655(02)00187-3.

    Parameters
    ----------
    candidates : list[T]
        List of candidates for a neighbor. Referred to as the set of all points $M$.
    distance : Callable[[T, T], float]
        Distance metric between candidates T, referred to as $d$ for a metric space.
    num_bases : int, optional
        To limit the number of inter-candidate distance calculated, we only compute the
        inter-candidate distance distances for the bases in the preprocessing. Selecting
        `num_bases` of bases is done by maximizing the distances between, by default 25
    """

    def __init__(self, candidates: list[T], distance: Callable[[T, T], float], num_bases: int=25):
        self.dist = distance
        self.candidates = candidates
        self.num_candidates = len(candidates)
        self.num_bases = num_bases

        # Used for LAESA aglorithim, we compute the distance between every point to the
        # base candidates so we can narrow our search with the traingle inequality
        self.base_indices = [random.choice(range(self.num_candidates))]  # arbitrary
        self.base_dist = [[0 for _ in range(self.num_candidates)] for _ in range(num_bases)]
        lower_bounds = [0 for _ in range(self.num_candidates)]

        for i in range(num_bases):
            current_base = candidates[self.base_indices[i]]
            max_dist_index = 0

            for j in range(self.num_candidates):  # TODO this step is parallelizable
                self.base_dist[i][j] = self.dist(current_base, candidates[j])
                if j in self.base_indices or self.base_indices[i] == j:  # d(x, x) = 0
                    continue

                lower_bounds[j] += self.base_dist[i][j]
                # We want the next base to be as far from the others as possible
                # ensures we have a diversity of bases
                max_dist_index = max(j, max_dist_index, key=lambda k: lower_bounds[k])

            self.base_indices.append(max_dist_index)
        self.base_indices.pop()  # Removes last base as we don't compute distances

    def predict(self, target: T) -> T:
        target_dist = [self.dist(target, self.candidates[p]) for p in self.base_indices]  # TODO parellize

        # Computes initial guess lower bounds  TODO this step is parallelizable
        def compute_lb(j: int) -> float:
            """Computes highest lb using the triangle inequality and the bases."""
            return max(abs(target_dist[i] - self.base_dist[i][j]) for i in range(self.num_bases))
        lower_bounds = [compute_lb(j) for j in range(self.num_candidates)]

        base_index = min(range(self.num_bases), key=lambda i: target_dist[i])
        best_dist = target_dist[base_index]
        best_candidate = self.base_indices[base_index]

        # We assume our lowerbounds total ordering is approximately correct
        # The heap ensures that all further lower bounds are greater than the best dist
        # Heapify is O(n) and this value should converge in O(1) steps
        lb_heap = Heap(range(self.num_candidates), key=lambda i: lower_bounds[i])
        while lb_heap and lower_bounds[lb_heap.peak()] <= best_dist:
            cand_index = lb_heap.pop()
            if (new_dist := self.dist(self.candidates[cand_index], target)) < best_dist:
                best_dist, best_candidate = new_dist, cand_index

        return self.candidates[best_candidate]


class Aesa[T]:
    """Computes approximate Nearest Neighbor with linear preprocessing.

    References
    ----------
    .. [1] E. Vidal Ruiz,
    “An algorithm for finding nearest neighbours in (approximately) constant average time,”
    Pattern Recognition Letters, vol. 4, no. 3, pp. 145-157, Jul. 1986, doi: 10.1016/0167-8655(86)90013-9.

    Parameters
    ----------
     candidates : list[T]
        List of candidates for a neighbor. Referred to as the set of all points $M$.
    distance : Callable[[T, T], float]
        Distance metric between candidates T, referred to as $d$ for a metric space.
    """
    def __init__(self, candidates: list[T], distance: Callable[[T, T], float]):
        self.candidates = candidates
        self.dist = distance
        self.num_candidates = len(candidates)
        self.cand_dist = [[distance(x, y) for x in candidates] for y in candidates]

    def predict(self, target: T) -> T:
        best_candidate = 0
        best_dist = float("inf")
        lower_bounds = [0] * self.num_candidates
        alive = list(range(self.num_candidates))

        while alive:
            active = min(alive, key=lambda i: lower_bounds[i])
            active_dist = self.dist(self.candidates[active], target)

            if active_dist < best_dist:
                best_candidate = active
                best_dist = active_dist

            new_alive = []
            for i in alive:
                # Compute the lower bound relative to the active candidate
                lower_bound = abs(active_dist - self.cand_dist[active][i])
                # Use the highest lower bound overall for this candidate
                lower_bounds[i] = max(lower_bounds[i], lower_bound)
                # Check if this candidate remains alive
                if lower_bounds[i] < best_dist:
                    new_alive.append(i)

            alive = new_alive

        return self.candidates[best_candidate]

import numpy as np
import tqdm

def dist(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

if __name__ == "__main__":  # TODO implement LAESA KNN
    res = []
    data = list(np.random.randint(0, 10, (100, 10)))
    a = Laesa(data, dist)
    targ = np.random.randint(0, 100, (10))
    print(targ)
    print(a.predict(data))
    print(min(data, key=lambda i: dist(i, targ)))


    print(f"dist pred: {dist(targ, a.predict(targ))}, dist actual: {min(dist(i, targ) for i in data)}")
