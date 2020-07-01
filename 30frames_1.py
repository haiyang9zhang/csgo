#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import os
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker1 import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

warnings.filterwarnings('ignore')


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = 'video.mp4'

    # load the COCO class labels our YOLO model was trained on    加载我们的YOLO模型经过培训的COCO类标签
    labelsPath = os.path.sep.join(["model_data", "coco_classes.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # print(str(len(LABELS))+"load successfully")
    # print(LABELS)
    class_nums = np.zeros(80)
    counter = {}
    if asyncVideo_flag:
        video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('30frames_1.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    frame_cnt = 0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True or frame_cnt > 30:
            break

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, confidence, class_names = yolo.detect_image(image)

        features = encoder(frame, boxs)

        detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in
                      zip(boxs, confidence, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        # print("print indices!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(indices)

        # for i in indices:
        #    print(str(i)+class_names[i][0])

        # print(indices.shape)
        class_last_nums = class_nums.copy()

        detections = [detections[i] for i in indices]
        class_names = [class_names[i] for i in indices]
        print("class_name:" + str(class_names))

        class_IDs = []
        current_nums = np.zeros(80)
        # class_IDs=[]
        for class_name in class_names:
            for i, LABEL in enumerate(LABELS):
                if class_name[0] == LABEL:
                    current_nums[i] += 1
                    class_IDs.append(i)
        # print("person:"+str(current_nums[0]))

        cv2.putText(frame, 'Current', (50, 60), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        x1 = 50
        y1 = 100
        for i, cl in enumerate(current_nums):
            if cl > 0:
                cv2.putText(frame, LABELS[i] + "=" + str(cl), (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255),
                            2)  # 当前帧各类别数量
                y1 = y1 + 20

        for i, det in enumerate(detections):
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 1)
            # cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0, 255, 0), 2)
            # cv2.putText(frame, class_names[i],(int(bbox[0]), int(bbox[1])-5), 0, 5e-3 * 130, (0, 255, 0), 2)
            # cv2.putText(frame, class_names[i], (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 130,(0, 255, 255), 2)
            cv2.putText(frame, class_names[i][0], (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0, 255, 0), 2)

        print("Total of detections:" + str(len(detections)))
        # Call the tracker
        tracker.predict()
        tracker.update(detections, class_IDs)

        # for i, cl in enumerate(class_nums):
        #     if cl > 0:
        #         print("add: " + LABELS[i] + str(cl - class_last_nums[i]))

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                print("not track.is_confirmed() or track.time_since_update > 1: " + str(track.track_id))
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
            counter[track.class_id].append(track.track_id)
            # class_nums[track.class_id].append(track.track_id)
            # print(str(LABELS[track.class_id])+":"+class_nums[track.class_id])
            # print("track.id: " + str(track.track_id))
            # print("track.class_name: " + str(LABELS[track.class_id]))

        print(str(counter))
        print("--------------------------该帧输出完毕！--------------------------------------")

        # cv2.putText(frame, 'Total', (200, 60), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 0), 2)
        # x2 = 200
        # y2 = 100
        # for i, cl in enumerate(class_nums):
        #     if cl > 0:
        #         cv2.putText(frame, LABELS[i] + "=" + str(cl), (x2, y2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 2)
        #         y2 = y2 + 20

        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)  # ！！！！！！！！！输出FPS
        # cv2.imshow('', frame)

        if writeVideo_flag:  # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        fps = (fps + (1. / (time.time() - t1))) / 2

        # print("FPS = %f"%(fps))

        frame_cnt += 1
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())