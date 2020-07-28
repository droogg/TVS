import time
import torch
import warnings
from .deep_sort import build_tracker


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.video_path = video_path

        use_cuda = args['use_cuda'] and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self, idx_frame, frame_interval, im, bbox_xywh, cls_conf):
        if idx_frame % frame_interval:
            pass
        else:
            start = time.time()
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
            end = time.time()
            print('time:', end-start)
            return outputs
        return []