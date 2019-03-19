from .dataset import VideoDataset,ProcessDesc
from .loader import VideoLoader
from .rnb_dataset import RnBDataset
from .rnb_loader import RnBLoader, Sampler

def video_size_from_file(filename):
    return lib.nvvl_video_size_from_file(str.encode(filename))
