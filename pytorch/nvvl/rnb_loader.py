import collections
import torch
import time
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
        self.consecutive_frames = consecutive_frames
        self.frames_indices_queue = collections.deque()



    def _receive_batch(self):
        batch_size = self.batch_size_queue.popleft()
        frame_indices = self.frames_indices_queue.popleft()

        t = self.dataset._create_tensor_map(batch_size)
        labels = []
        for i in range(batch_size):
            _, label = self.dataset._start_receive(t, i)
            labels.append(label)
        
        t_input = t['input']
        #if overlapping frames existed in input video frame request, reformat tensors
        if batch_size != len(frame_indices):
            new_t = self.dataset._create_tensor_map(len(frame_indices))
            for i in range(len(frame_indices)):
                quotient = i // self.consecutive_frames
                remainder = i % self.consecutive_frames
                if remainder == 0:
                    new_t['input'][i] = t_input[quotient]
                else:
                    new_t['input'][i,0:self.consecutive_frames-remainder] = t_input[quotient, remainder:self.consecutive_frames]
                    new_t['input'][i,self.consecutive_frames-remainder:self.consecutive_frames]=t_input[quotient+1, 0:remainder]
            self.tensor_queue.append((batch_size, new_t, labels))
        else:
            self.tensor_queue.append((batch_size, t, labels))


    def loadfile(self, filename):
        length = self.dataset.get_length(filename)
        frame_indices = self.sampler.sample(length)
        self.frames_indices_queue.append(frame_indices)

        #if overlapping frames exist, reformat frame_indices to not load overlapping frames
        frames_set = []
        if frame_indices[1] < frame_indices[0] + self.consecutive_frames:
            frames_set = [i for i in range(frame_indices[0], frame_indices[len(frame_indices)-1]+self.consecutive_frames)]
            import math
            frame_indices = [i*self.consecutive_frames+1 for i in range(0, math.ceil(len(frames_set)/self.consecutive_frames))]
        self.dataset._read_file(filename, frame_indices)
        self.batch_size_queue.append(len(frame_indices))


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
        
        return t['input']

    def __iter__(self):
        return self


    def flush(self):
        self.dataset.close_all_files()


    def close(self):
        self.dataset.close()
