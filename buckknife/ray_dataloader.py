"""Make PyTorch dataloading faster with Ray (sometimes)."""

from typing import Optional, Dict

from warnings import warn

import ray

from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter, _DatasetKind
from torch.utils.data import _utils


class RayDataLoader(DataLoader):
    """
    Do CPU-bound data loading work on a Ray cluster.

    Motivation: when training deep models, data preparation is often done on the CPU while the model
    itself is trained on the GPU. In our academic-scale cluster (where a ratio of 4-20 CPU cores per GPU
    is common) this can pose a problem. This package aims to solve this transparently by distributing
    the CPU-bound work across a Ray cluster, which can increase that ratio -- limited only by the total
    number of available CPU cores the user can allocate across the cluster.

    The behaviour is intended to be very similar to a `DataLoader` with `num_workers > 0`.
    Instead of parallelizing across processes on the same node, the tasks are sent to a Ray
    cluster, if one is available. If one isn't available, it falls back to default DataLoader
    behaviour.

    Specifically, the preparation of each _batch_ is submitted as a Ray task. Some caveats:

    * If preparing each batch is quick relative to how expensive it is to move the data across
      nodes or processes, this will likely be less efficient than the default implementation.
    * If your `Dataset` is dependent on something like a filesystem, this filesystem must be
      accessible in the Python process representing the Ray worker the task gets assigned to.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Params:
            ray_remote_args: A Dict of kwargs to be passed to `.options` on the Ray task. Example:
                             {"num_cpus": 1, "name": "dataset_fetch_task"}.
            prefetch_factor: The maximum number of batch-fetching tasks to schedule in advance.
            *args: Passed to `DataLoader` constructor.
            **kwargs: Passed to `DataLoader` constructor.
        """
        self._ray_remote_args = kwargs.pop("ray_remote_args", {})
        self.ray_prefetch_factor = kwargs.pop("prefetch_factor", 1)

        super().__init__(*args, **kwargs)

    def _get_iterator(self) -> _BaseDataLoaderIter:
        if ray.is_initialized():
            return _RayDataLoaderIter(self, self._ray_remote_args)

        warn("Ray has not been initialized. Falling back to the base DataLoader implementation.")
        return super()._get_iterator()


class _RayDataLoaderIter(_BaseDataLoaderIter):
    """
    Key logic:
    
    * On instantiation, `ray.put` the `dataset_fetcher` (which contains the dataset). This could get
      ugly if the dataset instance is big. (Hint: instead of an in-memory cache, maybe use the Ray
      object store for caching the result of `__getitem__`).
    * When fetching the next batch:
        * Make sure up to `prefetch_factor` batch-fetching tasks are scheduled
        * Call `ray.get` on the next task, blocking until ready.
        * If pinning memory, obey that.

    In theory, everything else should be the same as a regular data loader.
    """

    def __init__(self, loader, remote_args: Dict):
        super().__init__(loader)

        self._prefetch_factor = loader.ray_prefetch_factor
        assert self._prefetch_factor > 0, "prefetch_factor must be >0"

        self.dataset_fetcher = ray.put(
            _DatasetKind.create_fetcher(
                self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last
            )
        )
        self._fetch = _fetch.options(**remote_args)
        self._prefetched = []
        self._cease_prefetching = False

    def _next_data(self):
        while (not self._cease_prefetching) and (len(self._prefetched) < self._prefetch_factor):
            try:
                index = self._next_index()
            except StopIteration:
                self._cease_prefetching = True
                break
            self._prefetched.append(
                self._fetch.remote(self.dataset_fetcher, index)
            )

        data_pointer = self._prefetched.pop(0)
        try:
            data = ray.get(data_pointer)
        except StopIteration:
            self._cease_prefetching = True
            raise

        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

        return data


@ray.remote
def _fetch(dataset_fetcher, index):
    return dataset_fetcher.fetch(index)
