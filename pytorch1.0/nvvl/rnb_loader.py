import collections
import torch
from .rnb_dataset import RnBDataset
from .dataset import ProcessDesc

class Sampler(object):
    def sample(self, length):
        raise NotImplementedError()


class AllSampler(Sampler):
    def __init__(self):
        pass

    def sample(self, length):
        return range(length)


class RnBLoader(object):
    def __init__(self, width, height, consecutive_frames, device_id=0, sampler=None):
        processing = {
            'input': ProcessDesc(scale_width=width, scale_height=height, random_flip=False)
        }
        self.dataset = RnBDataset(processing=processing, device_id=device_id, sequence_length=consecutive_frames)
        self.sampler = sampler if sampler is not None else AllSampler()

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

    def get_stats(self):
        return self.dataset.get_stats()

    def reset_stats(self):
        return self.dataset.reset_stats()

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        if not self.tensor_queue:
            assert self.dataset.samples_left == 0, "Tensor queue is empty but there are samples left in the VideoDataset"
            raise StopIteration

        # first fire off a receive to keep the pipe filled
        if self.batch_size_queue:
            self._receive_batch()

        batch_size, t, labels = self.tensor_queue.popleft()
        for i in range(batch_size):
            self.dataset._finish_receive()

        if any(label is not None for label in labels):
            t["labels"] = labels

        return t

    def __iter__(self):
        if self.dataset.samples_left != 0:
            raise RuntimeError("Need to exhaust iterator before creating a new one")

        for b in iter(self.batch_sampler):
            for i in b:
                self.dataset._read_sample(i)
            self.batch_size_queue.append(len(b))

        for i in range(self.buffer_length):
            self._receive_batch()
        return self
