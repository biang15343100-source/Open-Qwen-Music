
from __future__ import annotations

import numpy as np


class UnionFind:
    __slots__ = ("_parent", "_rank", "_components")

    def __init__(self, size: int) -> None:
        self._parent = np.arange(size, dtype=np.int32)
        self._rank = np.zeros(size, dtype=np.int8)
        self._components = size

    def find(self, node: int) -> int:
        parent = self._parent
        root = node
        while parent[root] != root:
            root = int(parent[root])

        while parent[node] != root:
            parent[node], node = root, int(parent[node])
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        rank = self._rank
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        self._components -= 1
        return True

    def union_sorted_groups(self, order: np.ndarray, keys: np.ndarray) -> int:
        if order.size < 2:
            return 0
        same = keys[1:] == keys[:-1]
        merged = 0
        for idx in np.flatnonzero(same):
            if self.union(int(order[idx]), int(order[idx + 1])):
                merged += 1
        return merged

    def roots(self) -> np.ndarray:
        out = np.empty(self._parent.size, dtype=np.int32)
        for i in range(self._parent.size):
            out[i] = self.find(i)
        return out

    @property
    def n_components(self) -> int:
        return self._components

    def __len__(self) -> int:
        return int(self._parent.size)
