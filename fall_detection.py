import cv2 as cv
import numpy as np
import math
from timeit import default_timer as timer
from train_model import create_model, WEIGHTS_PATH
from keras.applications.mobilenet_v2 import preprocess_input

font = cv.FONT_HERSHEY_SIMPLEX
green = (0, 255, 0)
red = (0, 0, 255)
line_type = cv.LINE_AA
IMAGE_SIZE = 224
MHI_DURATION = 1500 # milliseconds
THRESHOLD = 32
GAUSSIAN_KERNEL = (3, 3)

def start_fall_detector_FDD(video_path, annotation_path):
    ''' Uses the annotated frames from the FDD dataset for spatial stream '''
    model = create_model(WEIGHTS_PATH)
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print("Cannot open video {}".format(video_path))
        return

    annotation_file = open(annotation_path, "r")
    annotation_file.readline() # Skip line 1
    annotation_file.readline() # Skip line 2

    cv.namedWindow("Capture")
    cv.namedWindow("Cropped")
    cv.namedWindow("MHI")
    cv.moveWindow("Capture", 100, 100)
    cv.moveWindow("Cropped", 500, 100)
    cv.moveWindow("MHI", 800, 100)

    fps = int(video.get(cv.CAP_PROP_FPS))
    interval = int(max(1, math.ceil(fps/10) if (fps/10 - math.floor(fps/10)) >= 0.5 else math.floor(fps/10)))
    ms_per_frame = 1000 / fps # milliseconds
    count = interval

    prev_mhi = [np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32) for i in range(interval)]
    prev_timestamp = [i * ms_per_frame for i in range(interval)]
    prev_frames = [None] * interval
    for i in range(interval):
        ret, frame = video.read()
        frame = cv.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv.INTER_AREA)
        frame = cv.GaussianBlur(frame, GAUSSIAN_KERNEL, 0)
        prev_frames[i] = frame.copy()

    fall_frames_seen = 0 # Number of consecutive fall frames seen so far
    fall_detected = False
    MIN_NUM_FALL_FRAME = int(fps/5) # Need at least some number of frames to avoid flickery classifications

    while True:
        start_time = timer()
        ret, orig_frame = video.read()
        if not ret:
            break

        # Crop out bounding box
        annotations = annotation_file.readline().strip().split(",")
        x_start = int(annotations[2])
        y_start = int(annotations[3])
        x_end = int(annotations[4])
        y_end = int(annotations[5])
        if (x_start <= 0 and y_start <= 0 and x_end <= 0 and y_end <= 0):
            x_start = 0
            y_start = 0
            x_end = frame.shape[0]
            y_end = frame.shape[1]

        cropped = orig_frame[y_start:y_end, x_start:x_end].copy()
        try:
            cropped = cv.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv.INTER_LINEAR)
        except:
            cropped = cv.resize(orig_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv.INTER_LINEAR)

        labelled_frame = orig_frame.copy()
        cv.rectangle(
            labelled_frame, (x_start, y_start), (x_end, y_end),
            color = green, lineType = line_type
        )

        # Create MHI
        prev_ind = count % interval
        prev_timestamp[prev_ind] += interval * ms_per_frame
        count += 1

        frame = cv.resize(orig_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv.INTER_AREA)
        frame = cv.GaussianBlur(frame, GAUSSIAN_KERNEL, 0)
        frame_diff = cv.absdiff(frame, prev_frames[prev_ind])
        gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
        _, motion_mask = cv.threshold(gray_diff, THRESHOLD, 1, cv.THRESH_BINARY)

        cv.motempl.updateMotionHistory(motion_mask, prev_mhi[prev_ind], prev_timestamp[prev_ind], MHI_DURATION)
        mhi = np.uint8(np.clip((prev_mhi[prev_ind] - (prev_timestamp[prev_ind] - MHI_DURATION))/MHI_DURATION, 0, 1) * 255)
        prev_frames[prev_ind] = frame.copy()

        # Prepare input
        spatial_input = cropped.copy().astype(np.float32)
        spatial_input = cv.cvtColor(spatial_input, cv.COLOR_BGR2RGB)
        spatial_input = np.array([spatial_input])
        temporal_input = mhi.copy().astype(np.float32)
        temporal_input = cv.cvtColor(temporal_input, cv.COLOR_GRAY2RGB)
        temporal_input = np.array([temporal_input])
        preprocess_input(spatial_input)
        preprocess_input(temporal_input)

        # Make prediction
        prediction = np.round(model.predict([spatial_input, temporal_input]))[0]
        is_fall = prediction == 1
        if is_fall:
            fall_frames_seen = min(fall_frames_seen + 1, MIN_NUM_FALL_FRAME)
        else:
            fall_frames_seen = max(fall_frames_seen - 1, 0)

        if fall_frames_seen == MIN_NUM_FALL_FRAME:
            fall_detected = True
        elif fall_frames_seen == 0:
            fall_detected = False

        cv.putText(
            labelled_frame, "Status: {}".format("Fall detected!" if fall_detected else "No fall"), (5, 20),
            fontFace = font, fontScale = 0.5, color = red if fall_detected else green, lineType = line_type
        )

        # Show images
        cv.imshow("Capture", labelled_frame)
        cv.imshow("Cropped", cropped)
        cv.imshow("MHI", mhi)

        # Compensate for elapsed time used to process frame
        wait_time = int(max(1, ms_per_frame - (timer() - start_time) * 1000))
        if cv.waitKey(wait_time) == 27:
            break

    video.release()
    cv.destroyAllWindows()
    annotation_file.close()

def start_fall_detector_realtime(input_path = 0):
    ''' Capture RGB and MHI in real time and feed into model '''
    model = create_model(WEIGHTS_PATH)
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print("Cannot open video/webcam {}".format(input_path))
        return

    MHI_DURATION_SHORT = 300 # Uses for putting bounding box on recent motion

    fps = int(cap.get(cv.CAP_PROP_FPS))
    cap_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    interval = int(max(1, math.ceil(fps/10) if (fps/10 - math.floor(fps/10)) >= 0.5 else math.floor(fps/10)))
    ms_per_frame = 1000 / fps # milliseconds
    count = interval

    prev_mhi = [np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32) for i in range(interval)]
    prev_mhi_short = [np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)] * interval
    prev_timestamp = [i * ms_per_frame for i in range(interval)]
    prev_frames = [None] * interval
    for i in range(interval):
        ret, frame = cap.read()
        frame = cv.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv.INTER_AREA)
        frame = cv.GaussianBlur(frame, GAUSSIAN_KERNEL, 0)
        prev_frames[i] = frame.copy()

    fall_frames_seen = 0 # Number of consecutive fall frames seen so far
    fall_detected = False
    MIN_NUM_FALL_FRAME = int(fps/5) # Need at least some number of frames to avoid flickery classifications

    cv.namedWindow("Capture")
    cv.namedWindow("Cropped")
    cv.namedWindow("MHI")
    cv.moveWindow("Capture", 100, 100)
    cv.moveWindow("Cropped", 500, 100)
    cv.moveWindow("MHI", 800, 100)

    while True:
        start_time = timer()
        ret, orig_frame = cap.read()
        if not ret:
            break

        # Create MHI
        prev_ind = count % interval
        prev_timestamp[prev_ind] += interval * ms_per_frame
        count += 1

        frame = cv.resize(orig_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv.INTER_AREA)
        frame = cv.GaussianBlur(frame, GAUSSIAN_KERNEL, 0)
        frame_diff = cv.absdiff(frame, prev_frames[prev_ind])
        gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
        _, motion_mask = cv.threshold(gray_diff, THRESHOLD, 1, cv.THRESH_BINARY)
        prev_frames[prev_ind] = frame.copy()

        cv.motempl.updateMotionHistory(motion_mask, prev_mhi[prev_ind], prev_timestamp[prev_ind], MHI_DURATION)
        cv.motempl.updateMotionHistory(motion_mask, prev_mhi_short[prev_ind], prev_timestamp[prev_ind], MHI_DURATION_SHORT)
        mhi = np.uint8(np.clip((prev_mhi[prev_ind] - (prev_timestamp[prev_ind] - MHI_DURATION))/MHI_DURATION, 0, 1) * 255)
        mhi_short = np.uint8(np.clip((prev_mhi_short[prev_ind] - (prev_timestamp[prev_ind] - MHI_DURATION_SHORT))/MHI_DURATION_SHORT, 0, 1) * 255)

        # Crop out motion
        x_start = y_start = IMAGE_SIZE
        x_end = y_end = 0
        contours, _ = cv.findContours(mhi_short, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if (len(contours) > 0):
            for c in contours:
                contour = cv.approxPolyDP(c, 3, True)
                x, y, w, h = cv.boundingRect(contour)
                if x < x_start:
                    x_start = x
                if y < y_start:
                    y_start = y
                if x + w > x_end:
                    x_end = x + w
                if y + h > y_end:
                    y_end = y + h
        else:
            x_start = y_start = 0
            x_end = y_end = IMAGE_SIZE

        x_start = int(np.round(x_start / IMAGE_SIZE * cap_width))
        y_start = int(np.round(y_start / IMAGE_SIZE * cap_height))
        x_end = int(np.round(x_end / IMAGE_SIZE * cap_width))
        y_end = int(np.round(y_end / IMAGE_SIZE * cap_height))
        labelled_frame = orig_frame.copy()
        cv.rectangle(
            labelled_frame, (x_start, y_start), (x_end, y_end),
            color = green, lineType = line_type
        )
        cropped = orig_frame[y_start:y_end, x_start:x_end].copy()
        try:
            cropped = cv.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv.INTER_LINEAR)
        except:
            cropped = cv.resize(orig_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv.INTER_LINEAR)

        # Prepare input
        spatial_input = cropped.copy().astype(np.float32)
        spatial_input = cv.cvtColor(spatial_input, cv.COLOR_BGR2RGB)
        spatial_input = np.array([spatial_input])
        temporal_input = mhi.copy().astype(np.float32)
        temporal_input = cv.cvtColor(temporal_input, cv.COLOR_GRAY2RGB)
        temporal_input = np.array([temporal_input])
        preprocess_input(spatial_input)
        preprocess_input(temporal_input)

        # Make prediction
        prediction = np.round(model.predict([spatial_input, temporal_input]))[0]
        is_fall = prediction == 1
        if is_fall:
            fall_frames_seen = min(fall_frames_seen + 1, MIN_NUM_FALL_FRAME)
        else:
            fall_frames_seen = max(fall_frames_seen - 1, 0)

        if fall_frames_seen == MIN_NUM_FALL_FRAME:
            fall_detected = True
        elif fall_frames_seen == 0:
            fall_detected = False

        cv.putText(
            labelled_frame, "Status: {}".format("Fall detected!" if fall_detected else "No fall"), (5, 20),
            fontFace = font, fontScale = 0.5, color = red if fall_detected else green, lineType = line_type
        )

        # Show images
        cv.imshow("Capture", labelled_frame)
        cv.imshow("Cropped", cropped)
        cv.imshow("MHI", mhi)

        # Compensate for elapsed time used to process frame
        wait_time = int(max(1, ms_per_frame - (timer() - start_time) * 1000))
        if cv.waitKey(wait_time) == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # start_fall_detector_FDD(
    #     "datasets/FDD/Coffee_room_01/Videos/video (5).avi",
    #     "datasets/FDD/Coffee_room_01/Annotation_files/video (5).txt"
    # )
    start_fall_detector_realtime("datasets/FDD/Home_02/Videos/video (40).avi")
    # start_fall_detector_realtime("datasets/URFD/Fall_videos/fall-15-cam0-rgb.avi")
    # start_fall_detector_realtime("demo/test3.avi")
