import cv2 as cv
import numpy as np
import mediapipe as mp
import math, os

mp_face_mesh = mp.solutions.face_mesh

# left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

right_iris = [474, 475, 476, 477]
left_iris = [469, 470, 471, 472]

l_h_left = [33] # right eye right most landmark
l_h_right = [133] # right eye left most landmark
r_h_left = [362] # left eye right most landmark
r_h_right = [263] # left eye left most landmark

def euclideanDistance(points1, points2):
    x1, y1 = points1.ravel()
    x2, y2 = points2.ravel()

    # 유클리디안 거리 계산
    euclidean_distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)

    return euclidean_distance

def irisPosition(iris_center, right_point, left_point):
    center_to_right_dist = euclideanDistance(iris_center, right_point)
    total_dist = euclideanDistance(right_point, left_point)
    ratio = center_to_right_dist/total_dist
    iris_position = ""
    if ratio <= 0.4:
        iris_position = "right"
    elif ratio > 0.4 and ratio <= 0.6:
        iris_position = "center"
    else:
        iris_position = "left"
    return iris_position, ratio

def eyeTracking(video_path):
    cap = cv.VideoCapture(video_path)

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
    min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                mesh_points=np.array([np.multiply([p.x,p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[left_iris])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[right_iris])

                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

                iris_position, ratio = irisPosition(center_right, mesh_points[r_h_right], mesh_points[r_h_left][0])
                print(iris_position, "{:.2f}%".format(ratio))

                # cv.putText(frame, f"Iris pos: {iris_position}, {ratio:.2f}", 
                #         (30,30), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv.LINE_AA)

            # cv.imshow('img', frame)
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break

    cap.release()

video_path = 'C:/Users/user/Desktop/labeling_tool/all video/'

if __name__ == "__main__":
    video_list = os.listdir(video_path)
    for i in video_list:
        eyeTracking(video_path + i)