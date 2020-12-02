import os
import numpy as np
import cv2

SKIP_N = 20
PATTERN_SHAPE = (9, 6)
N_PATTERNS = PATTERN_SHAPE[0] * PATTERN_SHAPE[1]


def check_video_end(ret):
    return not ret


def read_n_frame(video_cap, n):
    for i in range(n-1):
        ret, frame = video_cap.read()

        if check_video_end(ret):
            return not ret, frame

    ret, frame = video_cap.read()

    return not ret, frame


def make_chessboard_black(frame, show):
    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    imghsv[:, :, 2] = [[max(pixel - 40, 0) if pixel < 220 else min(pixel + 40, 255) for pixel in row] for row in imghsv[:, :, 2]]
    if show:
        cv2.imshow('contrast', cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR))
        cv2.imshow('init', frame)

    return imghsv


def connect_real_3dchessboard_with_2dimage(video_path, *, show=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((N_PATTERNS, 3), np.float32)
    objp[:,:2] = np.mgrid[:PATTERN_SHAPE[0], :PATTERN_SHAPE[1]].T.reshape(-1,2)

    world_3Dpoints = []
    img_2Dpoints = []
    frame_shape = []

    cap = cv2.VideoCapture(video_path)
    while(True):
        is_vid_end, frame = read_n_frame(cap, SKIP_N)
        if is_vid_end:
            break

        frame_init = make_chessboard_black(frame, show)
        gray = cv2.cvtColor(frame_init, cv2.COLOR_BGR2GRAY)
        frame_shape = gray.shape[::-1]

        is_chessboard_found, corners = cv2.findChessboardCorners(gray, PATTERN_SHAPE, None)
        if is_chessboard_found:
            world_3Dpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            img_2Dpoints.append(corners2)
            cv2.drawChessboardCorners(frame, PATTERN_SHAPE, corners2, is_chessboard_found)

        if show:
            cv2.imshow("Chess", frame)
            cv2.waitKey(0)

    cap.release()

    return world_3Dpoints[::2], img_2Dpoints[::2], frame_shape


def find_camera_parameters(video_path):
    world_3Dpoints, img_2Dpoints, frame_shape = connect_real_3dchessboard_with_2dimage(video_path)

    print(f"Video was ended \nFind Camera Parameters by N={len(world_3Dpoints)} frames with found chessboard")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_3Dpoints, img_2Dpoints, frame_shape, None, None)
    print(f"Ret \n{ret}\n\n"
          f"MTX \n{mtx}\n\n"
          f"distortion \n{dist}\n\n"
          #f"Rotation Vectors \n{rvecs}\n\n"
          #f"Translation Vectors \n{tvecs}\n\n\n"
    )

    return ret, mtx, dist, rvecs, tvecs


def illustrate_undistortion(video_path, camera_parameters):
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    h, w = frame.shape[:2]

    ret, mtx, dist, _, _ = camera_parameters
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    while(True):
        is_video_run, frame = cap.read()
        if not is_video_run:
            break

        frame_undist = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        frame_undist = frame_undist[y:y+h, x:x+w]

        cv2.imshow("Distortion", cv2.resize(frame, dsize=(w//3,h//3)))
        cv2.imshow("Undistortion", cv2.resize(frame_undist, dsize=(w//3,h//3)))
        cv2.waitKey(100)

    cap.release()
    return


if __name__ == '__main__':
    video_path = os.path.join("/home/nemiop/it_jim/geomagical/Calibration/videos/iPhone/chessboard_calibration", "iphone12pro.MOV")

    camera_parameters = find_camera_parameters(video_path)
    illustrate_undistortion(video_path, camera_parameters)

    cv2.destroyAllWindows()



