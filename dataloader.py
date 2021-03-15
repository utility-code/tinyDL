import numpy as np
import multiprocessing
import queue
from itertools import cycle


class DataLoader:
    def __init__(
            self, dataset, batch_size=64, num_workers=1, prefetch_batchs=2):
        #  super().__init__(dataset, batch_size)
        self.index = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_batchs = prefetch_batchs
        self.output_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = cycle(range(self.num_workers))  # what?
        self.cache = {}
        self.prefetch_index = 0

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=self.worker_fn, args=(
                    self.dataset, index_queue, self.output_queue)
            )
            worker.daemon = True  # what?
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.prefetch()

    def __iter__(self):
        self.index = 0
        self.cache = {}
        self.prefetch_index = 0
        self.prefetch()

        return self

    def __del__(self):
        try:
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:
                q.cancel_join_thread()
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index,
                         self.batch_size)  # what?
        return self.default_stacker([self.get() for _ in range(batch_size)])

    def default_stacker(self, batch):
        # Choose what DS to return based on type of input
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        if isinstance(batch[0], (int, float)):
            return np.array(batch)
        if isinstance(batch[0], (list, tuple)):
            return tuple(self.default_stacker(var) for var in zip(*batch))

    def worker_fn(self, dataset, index_queue, output_queue):
        # Add to output_queue from index_queue
        while True:
            try:
                index = index_queue.get(timeout=0)
            except queue.Empty:
                continue
            if index is None:
                break
            output_queue.put((index, dataset[index]))

    def prefetch(self):
        while(
            self.prefetch_index < len(
                self.dataset) and self.prefetch_index < self.index + 2 * self.num_workers * self.batch_size
        ):
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def get(self):
        self.prefetch()
        if self.index in self.cache:
            item = self.cache[self.index]
            del self.cache[self.index]
        else:
            while True:
                try:
                    (index, data) = self.output_queue.get(timeout=0)
                except queue.Empty:
                    continue
                if index == self.index:
                    item = data
                    break
                else:
                    self.cache[index] = data
        self.index += 1
        return item
