import cv2 as cv
import numpy as np
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

right_iris = [474, 475, 476, 477]
left_iris = [469, 470, 471, 472]

l_h_left = [33] # right eye right most landmark
l_h_right = [133] # right eye left most landmark
r_h_left = [362] # left eye right most landmark
r_h_right = [263] # left eye left most landmark

def cal_fop(center_point, left_point, right_point):
    cp = np.array(center_point)
    lp = np.array(left_point)
    rp = np.array(right_point)

    vector1 = lp - rp
    vector2 = cp - rp

    fop = vector1 * (np.dot(vector1, vector2) / np.dot(vector1, vector1))
    center_to_fop = fop - (vector1 / 2)
    output = np.dot(center_to_fop, center_to_fop) / np.dot(vector1, vector1)
    output = round(output * 100, 2)
    return output

def eyeTracking(video_path):
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
    min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        frame = video_path
            
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

            ratio1 = cal_fop(center_right, mesh_points[r_h_right][0], mesh_points[r_h_left][0])
            ratio2 = cal_fop(center_left, mesh_points[l_h_right][0], mesh_points[l_h_left][0])

            frame = cv.flip(frame, 1)

            cv.putText(frame, f'EYE RATIO - R: {ratio1} L: {ratio2}', (30,30), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, cv.LINE_AA)
            print(f'EYE RATIO - R: {ratio1} L: {ratio2}')
       
    return frame

if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    checking_str = '중앙'
    while True:
        key = cv.waitKey(33)
        
        if key == ord('q'):
            break
        elif key == ord('i'):
            checking_str = '좌측'
        elif key == ord('p'):
            checking_str = '우측'
        elif key == ord('o'):
            checking_str = '중앙'
        r, f = cap.read()
        if not r:
            break
        print(f'눈동자 비율({checking_str}) : ', end='')
        f = eyeTracking(f)
        cv.imshow('eye', f)
    cv.destroyAllWindows()