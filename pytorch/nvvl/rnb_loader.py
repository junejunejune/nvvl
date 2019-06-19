import collections
import torch
import time

class RnBSampler(object):
    def __init__(self):
        pass

    def sample(self, length):
        return range(length)


class RnBLoader(object):
    def __init__(self, dataset, sampler=None, batch_size=1, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler if sampler is not None else RnBSampler()

        self.tensor_queue = collections.deque()
        self.batch_size_queue = collections.deque()



    def _receive_batch(self):
        batch_size = self.batch_size_queue.popleft()
        t = self.dataset._create_tensor_map(batch_size)
        labels = []
        for i in range(batch_size):
            _, label = self.dataset._start_receive(t, i)
            labels.append(label)

        self.tensor_queue.append((batch_size, t, labels))


    def loadfile(self, filename):
        length = self.dataset.get_length(filename)
        frame_indices = self.sampler.sample(length)
        self.dataset._read_file(filename, frame_indices)
        for _ in frame_indices:
            self.batch_size_queue.append(1)


    def __next__(self):
        if not self.tensor_queue and self.dataset.samples_left == 0:
            raise StopIteration
        
        if self.batch_size_queue:
            self._receive_batch()


        batch_size, t, labels = self.tensor_queue.popleft()
        for i in range(batch_size):
            self.dataset._finish_receive()


        if any(label is not None for label in labels):
            t["labels"] = labels

        return t

    def __iter__(self):
        return self
