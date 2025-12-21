from typing import Callable, Iterator, List, TypeVar, Optional
import random
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.state import FlexState
from typing import cast


T_sample = TypeVar("T_sample")

class LengthPoolSortDataset(SavableDataset[T_sample]):
    """
    Local Pooling Length Sorting:
      - pool_size 개의 샘플을 누적한 뒤, key_fn(sample) 기준으로 정렬하여 순서대로 출력한다.
      - 남은 샘플이 pool_size보다 적은 경우, 그 잔여 부분도 다시 정렬하여 출력한다.
    """
    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        pool_size: int,
        key_fn: Callable[[T_sample], int],
        ascending: bool,
        worker_config: WorkerConfig,
        tail_shuffle: bool = True,
        shuffle_seed: Optional[int] = None,  # If None, use worker_config.global_seed
    ):
        super().__init__(worker_config=worker_config)
        assert pool_size > 0
        self.dataset = dataset
        self.pool_size = pool_size
        self.key_fn = key_fn
        self.ascending = ascending
        self.tail_shuffle = tail_shuffle
        base_seed = (
            shuffle_seed
            if shuffle_seed is not None
            else getattr(worker_config, "global_seed", 1230)
        )
        # Independent RNG, not polluting the global
        self._rng = random.Random(base_seed)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        pool: List[T_sample] = []
        for batch_idx, sample in enumerate(self.dataset):
            pool.append(sample)
            if len(pool) >= self.pool_size:
                pool.sort(key=self.key_fn, reverse=not self.ascending)
                shuffle_seed = 42 + batch_idx
                random.Random(shuffle_seed).shuffle(pool)
                # print(f"flush pool #{batch_idx // self.pool_size}, batch idx:{batch_idx}, first_len={self.key_fn(pool[0])}")
                for s in pool:
                    yield s
                pool.clear()
        if pool:
            pool.sort(key=self.key_fn, reverse=not self.ascending)
            if self.tail_shuffle:
                # Only shuffle the tail pool for reproducibility
                self._rng.shuffle(pool)
            for s in pool:
                yield s
            pool.clear()

    # ---- Abstract method implementation delegation ----
    def reset_state_own(self) -> None:
        if hasattr(self.dataset, "reset_state_own"):
            self.dataset.reset_state_own()
        
    def worker_has_samples(self) -> bool:
        return self.dataset.worker_has_samples()

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()

    def assert_can_restore(self) -> None:
        self.dataset.assert_can_restore()

    def restore_sample(self, index):
        return self.dataset.restore_sample(index)

    def save_state(self) -> FlexState:
        return cast(
            FlexState,
            {
                "dataset": self.dataset.save_state(),
                "rng_state": self._rng.getstate(),
            },
        )

    def merge_states(self, states):
        return self.dataset.merge_states(states)

    def restore_state(self, state: FlexState) -> None:
        assert isinstance(state, dict)
        self.dataset.restore_state(state["dataset"])
        self._rng.setstate(state["rng_state"])

    def config(self):
        return {
            "type": type(self).__qualname__,
            "pool_size": self.pool_size,
            "ascending": self.ascending,
            "tail_shuffle": self.tail_shuffle,
            "dataset": self.dataset.config(),
        }
