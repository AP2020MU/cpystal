"""`data_structure`: General data structures.

"""
from __future__ import annotations

from typing import Any, Dict, List, TypeVar

import numpy as np
import numpy.typing as npt




class SemimutableDict(Dict[Any, Any]):
    """Semi-mutable dictionary inherited from `dict`

    The only difference from `dict` is that using `[]` is not allowed, but using `update_force` method is allowed to replace the value.
    """
    def __init__(self, *args: Any) -> None:
        super().__init__(args)
        self.__updatable: bool = False

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self and not self.__updatable:
            raise TypeError(f"elements of '{self.__class__.__name__}' cannot be changed by '[]' operator; use 'update_force' method")
        super().__setitem__(key, value)
        self.__updatable = False

    def update_force(self, key: Any, value: Any) -> None:
        """Instance method for replacing the value.

        Args:
            key (Any): Immutable object.
            value (Any): New value of `key`.
        """
        self.__updatable = True
        self[key] = value
        
class UnionFind:
    def __init__(self, n: int): # O(n)
        self.parent: list[int] = [-1]*n
        self.n: int = n
    
    def root(self, x: int) -> int:
        """Return the number of the root of `x`.

        Note:
            Time complexity is O(α(n)), 
            where n is the size of the entire set
            and α(n) is the inverse Ackermann function.
            This function satisfies the relation:
                A(α(n)-1, α(n)-1) < n <= A(α(n), α(n)).

        Args:
            x (int): The number of the element.

        Returns:
            int: The number of the root of `x`.
        """
        if self.parent[x] < 0:
            return x
        else:
            self.parent[x] = self.root(self.parent[x])
            return self.parent[x]
        
    def size(self, x: int) -> int:
        """Return the size of the group to which `x` belongs.

        Note:
            Time complexity is O(n), where n is the size of the entire set.

        Args:
            x (int): The number of the element.

        Returns:
            int: The number of the size of the group to which `x` belongs.
        """
        return -self.parent[self.root(x)]
    
    def merge(self, x: int, y: int) -> bool: # xとyを結合する O(α(n))
        """Merge `x` and `y`.

        Note:
            Time complexity is O(α(n)), where n is the size of the entire set.

        Args:
            x (int): The number of the element.
            y (int): The number of the element.

        Returns:
            bool: Whether `x` and `y` belonged to the same group.
        """
        x = self.root(x)
        y = self.root(y)
        if x == y:
            return False
        if self.parent[x] > self.parent[y]: # for optimization
            x, y = y, x
        self.parent[x] += self.parent[y]
        self.parent[y] = x
        return True
    
    def issame(self, x: int, y: int) -> bool:
        """Judge whether `x` and `y` belong to the same group.

        Note:
            Time complexity is O(α(n)), where n is the size of the entire set.

        Args:
            x (int): The number of the element.
            y (int): The number of the element.

        Returns:
            bool: Whether `x` and `y` belong to the same group.
        """
        return self.root(x) == self.root(y)
    
    def family(self, x: int) -> list[int]:
        """Return the group of `x`.

        Note:
            Time complexity is O(n), where n is the size of the entire set.

        Args:
            x (int): The number of the element.

        Returns:
            list[int]: The group of `x`.
        """
        return [i for i in range(self.n) if self.issame(i, x)]
    
    def maximum(self) -> list[int]:
        """Return the group which has the maximum size among the groups.

        Note:
            Time complexity is O(n), where n is the size of the entire set.

        Returns:
            list[int]: The group which has the maximum size among the groups.
        """
        return self.family(self.parent.index(min(self.parent)))
    
    def all_root(self) -> list[int]:
        """Return the roots of the groups.

        Note:
            Time complexity is O(n), where n is the size of the entire set.

        Returns:
            list[int]: The roots of the groups.
        """
        return [i for i in range(self.n) if self.parent[i] < 0]
    
    def decompose(self) -> list[list[int]]:
        """Return the groups.

        Note:
            Time complexity is O(nα(n)), where n is the size of the entire set.

        Args:
            x (int): The number of the element.

        Returns:
            list[list[int]]: The groups.
        """
        return [self.family(i) for i in self.all_root()]


def main() -> None:
    pass
    return

if __name__ == "__main__":
    main()

