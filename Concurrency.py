from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(items: list[T], worker: Callable[[T], R], max_workers: int | None = None) -> list[R]:
  if not items:
    return []

  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    return list(executor.map(worker, items))
