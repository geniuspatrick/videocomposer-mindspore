import binascii
import datetime
import os
import subprocess
import time

import cv2
import numpy as np
from mvextractor.videocap import VideoCap


def rand_name(length=16, suffix=""):
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix
        name += suffix
    return name


def draw_motion_vectors(frame, motion_vectors):
    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
            cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.1)
            # cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 2, cv2.LINE_AA, 0, 0.2)
    return frame


def extract_motion_vectors(input_video, fps=4, dump=False, verbose=False, visual_mv=False):
    if dump:
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        for child in ["frames", "motion_vectors"]:
            os.makedirs(os.path.join(f"out-{now}", child), exist_ok=True)
    temp = rand_name()
    # tmp_video = f'{temp}_{input_video}'
    tmp_video = os.path.join(input_video.split("/")[0], f"{temp}" + input_video.split("/")[-1])
    videocapture = cv2.VideoCapture(input_video)
    frames_num = videocapture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_video = videocapture.get(cv2.CAP_PROP_FPS)
    # check if enough frames
    if frames_num / fps_video * fps > 16:  #
        fps = max(fps, 1)
    else:
        fps = int(16 / (frames_num / fps_video)) + 1
    ffmpeg_cmd = (
        f"ffmpeg -threads 8 -loglevel error -i {input_video} -filter:v fps={fps} -c:v mpeg4 -f rawvideo {tmp_video}"
    )

    if os.path.exists(tmp_video):
        os.remove(tmp_video)

    subprocess.run(args=ffmpeg_cmd, shell=True, timeout=120)

    cap = VideoCap()
    # open the video file
    ret = cap.open(tmp_video)
    if not ret:
        raise RuntimeError(f"Could not open {tmp_video}")

    step = 0
    times = []

    frame_types = []
    frames = []
    mvs = []
    mvs_visual = []
    # continuously read and display video frames and motion vectors
    while True:
        if verbose:
            print("Frame: ", step, end=" ")

        tstart = time.perf_counter()

        # read next video frame and corresponding motion vectors
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()

        tend = time.perf_counter()
        telapsed = tend - tstart
        times.append(telapsed)

        # if there is an error reading the frame
        if not ret:
            if verbose:
                print("No frame read. Stopping.")
            break

        frame_save = np.zeros(frame.copy().shape, dtype=np.uint8)  # *255
        if visual_mv:
            frame_save = draw_motion_vectors(frame_save, motion_vectors)

        # store motion vectors, frames, etc. in output directory
        dump = False
        if frame.shape[1] >= frame.shape[0]:
            w_half = (frame.shape[1] - frame.shape[0]) // 2
            if dump:
                cv2.imwrite(os.path.join("./mv_visual/", f"frame-{step}.jpg"), frame_save[:, w_half:-w_half])
            mvs_visual.append(frame_save[:, w_half:-w_half])
        else:
            h_half = (frame.shape[0] - frame.shape[1]) // 2
            if dump:
                cv2.imwrite(os.path.join("./mv_visual/", f"frame-{step}.jpg"), frame_save[h_half:-h_half, :])
            mvs_visual.append(frame_save[h_half:-h_half, :])

        h, w = frame.shape[:2]
        mv = np.zeros((h, w, 2))
        position = motion_vectors[:, 5:7].clip((0, 0), (w - 1, h - 1))
        mv[position[:, 1], position[:, 0]] = motion_vectors[:, 0:1] * motion_vectors[:, 7:9] / motion_vectors[:, 9:]

        step += 1
        frame_types.append(frame_type)
        frames.append(frame)
        mvs.append(mv)
        # mvs_visual.append(frame_save)
    if verbose:
        print("average dt: ", np.mean(times))
    cap.release()

    if os.path.exists(tmp_video):
        os.remove(tmp_video)

    return frame_types, frames, mvs, mvs_visual
