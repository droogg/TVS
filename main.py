# from ctypes import *
import cv2
import os
import numpy as np
from typing import Union
from deep_sort_pytorch.yolov3_deepsort import VideoTracker
from darknet import darknet
import time
from glob import glob
import os.path
from pathlib import Path
from sort import *

from collections import namedtuple
import datetime

bounding_boxes_and_ids = []
bbs_ids = namedtuple('Bbox_id', 'time id_frame bb_id')

netMain = None
metaMain = None
altNames = None


def model_init(configPath: str, weightPath: str, metaPath: str) -> Tuple[Any, Any]:
    '''
    Model initialization and store on CUDA-memory until the script completes
    :param configPath: путь к конфиг-файлу YOLO
    :param weightPath: путь к весам YOLO
    :param metaPath: путь к файлу .data YOLO
    :return: None
    '''
    global metaMain, netMain, altNames
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    return netMain, metaMain

def video_read(pipe: Union[str, int], camera_set: dict = None, savePath_out_vid: str = './out.avi') -> tuple:
    '''
    Создает объект VideoCapture для захвата видео и VideoWriter для записи результата
    :param pipe: video (path to video file) or webcam streams (pipe = 0) or custom streams set
    :param camera_set: dict: dict of cv2 sets for camera: expected keys: width, height, fps (позволяет установить
                                                        параметры камеры через API cv2)
    :param savePath_out_vid: путь куда будет сохранен видео-результат
    :return: tuple - (VideoCapture_object, (Path_save_video, VideoWriter_object))
    '''
    out = None
    cap = cv2.VideoCapture(pipe)
    if not camera_set is None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_set['width'])  # set camera property: width of frame
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_set['height'])  # set camera property: height of frame
        cap.set(cv2.CAP_PROP_FPS, camera_set['fps'])
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) % 100
    print(f'Video mod: {w}x{h} at {fps:.2f} FPS')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    if savePath_out_vid:  # if a video file with predictions is stored
        out = cv2.VideoWriter(
            savePath_out_vid, cv2.VideoWriter_fourcc(*"MJPG"),
            25.0 if camera_set is None else camera_set['fps'],
            (darknet.network_width(netMain), darknet.network_height(netMain)))
    return cap, (savePath_out_vid, out)


def inference_loop(cap: cv2.VideoCapture, out: tuple, tracker_mod: str, trackers_kwarg: dict,
                   show_out: bool = True) -> None:
    '''
    Цикл распознавания и трекинга
    :param cap: VideoCapture объект для захвата видео
    :param out: VideoWriter объект для записи итогового видео
    :param tracker_mod: задает режим трекера 'sort' или 'deepsort'
    :param trackers_kwarg: параметры трекера
    :param show_out: показывать результирующее видео в процессе распознавания
    :return: None
    '''
    save_out_vid, out = out
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    tracker, mot_tracker = _trackers(mod=tracker_mod, trackers_kwarg=trackers_kwarg)
    idx_frame = 0
    frame_interval = 1
    while True:
        prev_time = time.time()
        '''=====================================================================
            фективная функция посылающая сигнал синхронизации перед созданием фрейма'''
        sync_signal(port=None)
        '''====================================================================='''
        ret, frame_read = cap.read()
        if frame_read is None:
            break
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        idx_frame += 1
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())  # np.ndarray to IMAGE
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        track_kwarg = {'mot_tracker': mot_tracker, 'detections': detections,
                       'idx_frame': idx_frame,
                       'frame_interval': frame_interval,
                       'im': frame_resized,
                       'bbox_xywh': np.array([bbox_allinfo[2] for bbox_allinfo in detections]) if len(
                           detections) else np.empty((0, 4)),
                       'cls_conf': np.array([bbox_allinfo[1] for bbox_allinfo in detections]),
                       }
        track_bbs_ids = tracker(
            **track_kwarg)  # [[xmin, ymin, xmax, ymax, track_id],[xmin, ymin, xmax, ymax, track_id],...]
        add_bbs_ids(track_bbs_ids, idx_frame)
        '''================================================
            Определение глобальных координат объекта и их отправка 
            автопилоту'''
        global_bbs_coord = real_global_coord(track_bbs_ids)
        send_for_autopilot(global_bbs_coord, port=None)
        '''================================================'''
        image = cvDrawBoxes(frame_resized, track_bbs_ids)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if save_out_vid:
            out.write(image)
        # det_store
        print(1 / (time.time() - prev_time))
        if show_out:
            cv2.imshow('Demo', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if save_out_vid:
        out.release()


class TrackerModException(ValueError):
    def __init__(self, mod):
        message = f'Tracker have two mod options: "sort" and "deepsort". mod=="{mod}" is not define!'
        super().__init__(message)


def _trackers(mod: str, trackers_kwarg: dict) -> tuple:
    """
    :param mod: 'sort' или 'deepsort'
    :param trackers_kwarg: словарь параметров трекера
    :return: функция трекера, движок трекера
    """
    trackers_dict = {'sort': [_tracker_sort, Sort],
                     'deepsort': [_tracker_deepsort, VideoTracker]
                     }
    try:
        tracker, init_tracker = trackers_dict[mod]
        mot_tracker = init_tracker(**trackers_kwarg)
    except KeyError:
        raise TrackerModException(mod)
    return tracker, mot_tracker


def _tracker_sort(mot_tracker, detections, *args, **kwarg):
    det4sort = np.empty([0, 5])
    for dett in detections:
        x, y, w, h = dett[2][0], \
                     dett[2][1], \
                     dett[2][2], \
                     dett[2][3]
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        score = dett[1]
        det4sort_row = np.hstack([xmin, ymin, xmax, ymax, score])
        det4sort = np.vstack([det4sort, det4sort_row])
    track_bbs_ids = mot_tracker.update(det4sort)
    return track_bbs_ids


def _tracker_deepsort(mot_tracker, idx_frame, frame_interval, im, bbox_xywh, cls_conf, *args, **kwargs):
    track_bbs_ids = mot_tracker.run(idx_frame, frame_interval, im, bbox_xywh, cls_conf)
    return track_bbs_ids

def check_img_store():
    default_path = './additionally/test_img_1024'

    def sort_func(name):
        name_number_part = Path(name).stem.replace('CF', '').replace('_CROP_', '')
        if len(name_number_part) == 7:
            name_number_part = Path(name).stem.replace('CF', '').replace('_CROP_', '0')
        return int(name_number_part)

    sample_list = sorted(glob(default_path + '*.png'), key=sort_func)
    return sample_list

def performBatchDetect(net, mata, sample, thresh= 0.25, hier_thresh=.5, nms=.45, batch_size=8):
    # import cv2
    # import numpy as np
    # NB! Image sizes should be the same
    # You can change the images, yet, be sure that they have the same width and height

    # net = load_net_custom(configPath.encode('utf-8'), weightPath.encode('utf-8'), 0, batch_size)
    # meta = load_meta(metaPath.encode('utf-8'))

    # t1 = time.time()
    sample_list = check_img_store()
    s1, s2 = sample
    for s in [s1,s2]:
        img_samples = s
        image_list = [cv2.imread(k) for k in img_samples]
        pred_height, pred_width, c = image_list[0].shape
        net_width, net_height = (network_width(net), network_height(net))

        img_list = []
        for custom_image_bgr in image_list:
            custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
            custom_image = cv2.resize(
                custom_image, (net_width, net_height), interpolation=cv2.INTER_NEAREST)
            custom_image = custom_image.transpose(2, 0, 1)
            img_list.append(custom_image)

        arr = np.concatenate(img_list, axis=0)
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(net_width, net_height, c, data)
        t1 = time.time()

        batch_dets = network_predict_batch(net, im, batch_size, pred_width,
                                                    pred_height, thresh, hier_thresh, None, 0, 0)
        t2 = time.time()
        batch_boxes = []
        batch_scores = []
        batch_classes = []
        for b in range(batch_size):
            num = batch_dets[b].num
            dets = batch_dets[b].dets
            if nms:
                do_nms_obj(dets, num, meta.classes, nms)
            boxes = []
            scores = []
            classes = []
            for i in range(num):
                det = dets[i]
                score = -1
                label = None
                for c in range(det.classes):
                    p = det.prob[c]
                    if p > score:
                        score = p
                        label = c
                if score > thresh:
                    box = det.bbox
                    left, top, right, bottom = map(int,(box.x - box.w / 2, box.y - box.h / 2,
                                                box.x + box.w / 2, box.y + box.h / 2))
                    boxes.append((top, left, bottom, right))
                    scores.append(score)
                    classes.append(label)
                    boxColor = (int(255 * (1 - (score ** 2))), int(255 * (score ** 2)), 0)
                    cv2.rectangle(image_list[b], (left, top),
                              (right, bottom), boxColor, 2)
            cv2.imwrite(os.path.basename(img_samples[b]),image_list[b])

            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_classes.append(classes)
        free_batch_detections(batch_dets, batch_size)
        # t2 = time.time()
        print('time: ', t2-t1)
    return batch_boxes, batch_scores, batch_classes


def cvDrawBoxes(img, track_bbs_ids):
    for detection in track_bbs_ids:
        xmin, ymin, xmax, ymax, track_id = detection
        track_id = int(track_id)
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img, " [" + ' ID_' + str(track_id) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    [0, 255, 0], 2)
    return img


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def sync_signal(port=None):
    '''TODO: заменить заглушку на рабочий код'''
    pass


def add_bbs_ids(track_bbs_ids, idx_frame):
    bounding_boxes_and_ids.append(bbs_ids(str(datetime.datetime.now().time()), idx_frame, track_bbs_ids))


def get_real_global_center(port=None):
    '''TODO: заменить заглушку на рабочий код'''
    pass
    return 'center coord and additional info'


def real_global_coord(track_bbs_ids):
    '''TODO: заменить заглушку на рабочий код'''
    center_coord = get_real_global_center(port=None)
    '''математика для определения глобальных координат Bboxes'''
    return "global Bboxes coord's"


def send_for_autopilot(global_bbs_coord, port=None):
    '''TODO: заменить заглушку на рабочий код'''
    pass


if __name__ == "__main__":

    configPath = "./darknet/cfg/yolov4.cfg"  # in darknet dir
    # configPath = "./additionally/yolov4-cowc_carpk.cfg"
    weightPath = "./additionally/yolov4.weights"  # in darknet dir
    # weightPath = "./additionally/yolov4-cowc_carpk_best.weights"
    metaPath = "./additionally/coco.data"  # in darknet dir
    # metaPath = "./additionally/cowc_carpk.data"

    pipe = './additionally/test3.mpg'
    # tracker_mod = 'deepsort'
    tracker_mod = 'sort'
    if tracker_mod == 'sort':
        trackers_kwarg = {}
    elif tracker_mod == 'deepsort':
        from deep_sort_pytorch.utils.parser import get_config

        config_deepsort = './additionally/deep_sort.yaml'
        print(os.getcwd())
        cfg = get_config()
        cfg.merge_from_file(config_deepsort)
        trackers_kwarg = {
            'args': {'use_cuda': True, 'config_deepsort': config_deepsort},
            'cfg': cfg,
            'video_path': pipe,
        }

    # model_init(configPath=configPath, weightPath=weightPath, metaPath=metaPath)
    # cap, out = video_read(pipe=pipe, savePath_out_vid="./additionally/output.avi")
    # inference_loop(cap=cap, out=out, tracker_mod=tracker_mod, trackers_kwarg=trackers_kwarg, show_out=True)
    # print(bounding_boxes_and_ids)

    net, meta = model_init(configPath=configPath, weightPath=weightPath, metaPath=metaPath)

