import bisect
import cffi
import collections
import random
import sys
import torch
import torch.utils.data
import time

from . import lib
from .dataset import ProcessDesc, log_levels


class RnBDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length=1, device_id=0,
                 get_label=None, processing=None, log_level="warn"):
        self.ffi = lib._ffi
        self.sequence_length = sequence_length
        self.device_id = device_id
        self.get_label = get_label if get_label is not None else lambda x,y,z: None
        self.ts = []
        self.tts = []


        self.processing = processing
        if self.processing is None:
            self.processing = {"default" : ProcessDesc()}
        elif "labels" in processing and get_label is not None:
            raise KeyError("Processing must not have a 'labels' key when get_label is not None.")

        try:
            log_level = log_levels[log_level]
        except KeyError:
            print("Invalid log level", log_level, "using warn.", file=sys.stderr)
            log_level = lib.LogLevel_Warn

        if sequence_length < 1:
            raise ValueError("Sequence length must be at least 1")

        self.loader = lib.nvvl_create_video_loader_with_log(self.device_id, log_level)

        size = lib.nvvl_video_size(self.loader)
        self.width = size.width
        self.height = size.height

        for name, desc in self.processing.items():
            if desc.width == 0:
                if desc.scale_width == 0:
                    desc.width = self.width
                else:
                    desc.width = desc.scale_width

            if desc.height == 0:
                if desc.scale_height == 0:
                    desc.height = self.height
                else:
                    desc.height = desc.scale_height

            if desc.count == 0:
                desc.count = self.sequence_length


        self.samples_left = 0

        self.seq_queue = collections.deque()
        self.seq_info_queue = collections.deque()

        self.get_count = 0
        self.get_count_warning_threshold = 1000
        self.disable_get_warning = False

    def get_stats(self):
        return lib.nvvl_get_stats(self.loader)

    def reset_stats(self):
        return lib.nvvl_reset_stats(self.loader)

    def set_log_level(self, level):
        """Sets the log level from now forward

        Parameters
        ----------
        level : string
            The log level, one of "debug", "info", "warn", "error", or "none"
        """
        lib.nvvl_set_log_level(self.loader, log_levels[level])


    def get_length(self, filename):
        return lib.nvvl_frame_count(self.loader, str.encode(filename))


    def _read_file(self, filename, frame_indices):
        for index in frame_indices:
            lib.nvvl_read_sequence(self.loader, str.encode(filename),
                                   index, self.sequence_length)
            self.seq_info_queue.append((filename, index))
            self.samples_left += 1


    def _read_sample(self, index):
        # we want bisect_right here so the first frame in a file gets the file, not the previous file
        file_index = bisect.bisect_right(self.start_index, index)
        frame = index - self.start_index[file_index - 1] if file_index > 0 else index
        filename = self.filenames[file_index]

        lib.nvvl_read_sequence(self.loader, str.encode(filename),
                               frame, self.sequence_length)
        self.seq_info_queue.append((filename, frame))
        self.samples_left += 1

    def _get_layer_desc(self, desc):
        d = desc.desc()
        changes = {}

        if (desc.random_crop and (self.width > desc.width)):
            d.crop_x = random.randint(0, self.width - desc.width)
        else:
            d.crop_x = 0
        changes['crop_x'] = d.crop_x

        if (desc.random_crop and (self.height > desc.height)):
            d.crop_y = random.randint(0, self.height - desc.height)
        else:
            d.crop_y = 0
        changes['crop_y'] = d.crop_y

        if (desc.random_flip):
            d.horiz_flip = random.random() < 0.5
        else:
            d.horiz_flip = False
        changes['horiz_flip'] = d.horiz_flip

        return d[0], changes

    def _start_receive(self, tensor_map, index=0):
        tmp0 = time.time()

        seq = lib.nvvl_create_sequence_device(self.sequence_length, self.device_id)
        rand_changes = {}

        for name, desc in self.processing.items():
            tensor = tensor_map[name]
            layer = self.ffi.new("struct NVVL_PicLayer*")
            if desc.tensor_type == torch.cuda.FloatTensor:
                layer.type = lib.PDT_FLOAT
            elif desc.tensor_type == torch.cuda.HalfTensor:
                layer.type = lib.PDT_HALF
            elif desc.tensor_type == torch.cuda.ByteTensor:
                layer.type = lib.PDT_BYTE

            strides = tensor[index].stride()
            try:
                desc.stride.x = strides[desc.dimension_order.index('w')]
                desc.stride.y = strides[desc.dimension_order.index('h')]
                desc.stride.n = strides[desc.dimension_order.index('f')]
                desc.stride.c = strides[desc.dimension_order.index('c')]
            except ValueError:
                raise ValueError("Invalid dimension order")
            layer.desc, rand_changes[name] = self._get_layer_desc(desc)
            if desc.index_map_length > 0:
                layer.index_map = desc.index_map
                layer.index_map_length = desc.index_map_length
            else:
                layer.index_map = self.ffi.NULL
            layer.data = self.ffi.cast("void*", tensor[index].data_ptr())
            lib.nvvl_set_layer(seq, layer, str.encode(name))

        filename, frame = self.seq_info_queue.popleft()
        self.seq_queue.append(seq)
        tmp1 = time.time()
        lib.nvvl_receive_frames(self.loader, seq)
        # lib.nvvl_receive_frames_sync(self.loader, seq)
        tmp2 = time.time()
        self.ts.append(tmp1 - tmp0)
        self.tts.append(tmp2 - tmp1)
        return seq, self.get_label(filename, frame, rand_changes)

    def _finish_receive(self, synchronous=False):
        if not self.seq_queue:
            raise RuntimeError("Unmatched receive")

        if self.samples_left <= 0:
            raise RuntimeError("No more samples left in decoder pipeline")

        seq = self.seq_queue.popleft()
        # print(len(self.seq_queue))


        if synchronous:
            lib.nvvl_sequence_wait(seq)
        else:
            lib.nvvl_sequence_stream_wait_th(seq)

        lib.nvvl_free_sequence(seq)
        self.samples_left -= 1

    def _create_tensor_map(self, batch_size=1):
        tensor_map = {}
        with torch.cuda.device(self.device_id):
            for name, desc in self.processing.items():
                tensor_map[name] = desc.tensor_type(batch_size, *desc.get_dims())
        return tensor_map


    def summary(self):
        return self.ts, self.tts
    
    def close(self):
        lib.nvvl_destroy_video_loader(self.loader)

    def close_all_files(self):
        lib.nvvl_close_all_files(self.loader)
