import face_recognition
import numpy
from cv2 import cv2

#get vertical movement of the lip center
def movement_vertical_lip_center(previous_lip, lip):
    dist = numpy.zeros([6], dtype=float)

    #calculate the vertical movement of all points in the lip center
    #between the current frame and the previous one
    dist[0] = previous_lip[2][1] - lip[2][1]
    dist[1] = previous_lip[3][1] - lip[3][1]
    dist[2] = previous_lip[4][1] - lip[4][1]
    dist[3] = previous_lip[8][1] - lip[8][1]
    dist[4] = previous_lip[9][1] - lip[9][1]
    dist[5] = previous_lip[10][1] - lip[10][1]

    #calculate the average vertical movement of all points
    movement = sum(dist)/len(dist)
    return movement

#get vertical and horizontal movement of left lip corner
def movement_left_lip_corner(previous_top_lip, previous_bottom_lip, top_lip, bottom_lip):
    dist_x = numpy.zeros([4], dtype=float)
    dist_y = numpy.zeros([4], dtype=float)

    #calculate the horizontal movement of all points of the left lip corner
    #between the current frame and the previous one
    dist_x[0] = previous_top_lip[5][0] - top_lip[5][0]
    dist_x[1] = previous_top_lip[6][0] - top_lip[6][0]
    dist_x[2] = previous_top_lip[7][0] - top_lip[7][0]
    dist_x[3] = previous_bottom_lip[1][0] - bottom_lip[1][0]

    #calculate the vertical movement of all points of the left lip corner
    #between the current frame and the previous one
    dist_y[0] = previous_top_lip[5][1] - top_lip[5][1]
    dist_y[1] = previous_top_lip[6][1] - top_lip[6][1]
    dist_y[2] = previous_top_lip[7][1] - top_lip[7][1]
    dist_y[3] = previous_bottom_lip[1][1] - bottom_lip[1][1]

    #calculate the average horizontal and vertical movement of all points
    movement_x = sum(dist_x)/len(dist_x)
    movement_y = sum(dist_y)/len(dist_y)
    return movement_x, movement_y

#get vertical and horizontal movement of right lip corner
def movement_right_lip_corner(previous_top_lip, previous_bottom_lip, top_lip, bottom_lip):
    dist_x = numpy.zeros([4], dtype=float)
    dist_y = numpy.zeros([4], dtype=float)

    #calculate the horizontal movement of all points of the right lip corner
    #between the current frame and the previous one
    dist_x[0] = previous_top_lip[0][0] - top_lip[0][0]
    dist_x[1] = previous_top_lip[1][0] - top_lip[1][0]
    dist_x[2] = previous_top_lip[11][0] - top_lip[11][0]
    dist_x[3] = previous_bottom_lip[5][0] - bottom_lip[5][0]

    #calculate the vertical movement of all points of the right lip corner
    #between the current frame and the previous one
    dist_y[0] = previous_top_lip[0][1] - top_lip[0][1]
    dist_y[1] = previous_top_lip[1][1] - top_lip[1][1]
    dist_y[2] = previous_top_lip[11][1] - top_lip[11][1]
    dist_y[3] = previous_bottom_lip[5][1] - bottom_lip[5][1]

    #calculate the average horizontal and vertical movement of all points
    movement_x = sum(dist_x)/len(dist_x)
    movement_y = sum(dist_y)/len(dist_y)
    return movement_x, movement_y

#get vertical movement of the eyebrow
def movement_vertical_brow(previous_eyebrow, eyebrow):
    dist = numpy.zeros([5], dtype=float)

    #calculate the vertical movement of all points of the eyebrow
    #between the current frame and the previous one
    for point in range(len(dist)):
        dist[point] = previous_eyebrow[point][1] - eyebrow[point][1]

    #calculate the average vertical movement of all points
    movement = sum(dist)/len(dist)
    return movement

#get vertical movement of the left and right cheek
def movement_vertical_cheeks(previous_chin, chin):
    dist_left = numpy.zeros([3], dtype=float)
    dist_right = numpy.zeros([3], dtype=float)

    #calculate the vertical movement of all points of the left cheek
    #between the current frame and the previous one
    dist_left[0] = previous_chin[13][1] - chin[13][1]
    dist_left[1] = previous_chin[14][1] - chin[14][1]
    dist_left[2] = previous_chin[15][1] - chin[15][1]

    #calculate the vertical movement of all points of the right cheek
    #between the current frame and the previous one
    dist_right[0] = previous_chin[1][1] - chin[1][1]
    dist_right[1] = previous_chin[2][1] - chin[2][1]
    dist_right[2] = previous_chin[3][1] - chin[3][1]

    #calculate the average vertical movement of all points
    movement_left = sum(dist_left)/len(dist_left)
    movement_right = sum(dist_right)/len(dist_right)
    return movement_right, movement_left

#calculate how much the opening of the eye has changed
def eye_blinking(previous_eye, eye):
    current_dist = numpy.zeros([2], dtype=float)
    previous_dist = numpy.zeros([2], dtype=float)
    dist = numpy.zeros([2], dtype=float)

    #calculate distance of upper and lower eyelid for the current eye
    current_dist[0] = eye[5][1] - eye[1][1]
    current_dist[1] = eye[4][1] - eye[2][1]

    #calculate distance of upper and lower eyelid for the previous eye
    previous_dist[0] = previous_eye[5][1] - previous_eye[1][1]
    previous_dist[1] = previous_eye[4][1] - previous_eye[2][1]

    #calculate the difference between the current and the previous eye opening
    dist[0] = current_dist[0] - previous_dist[0]
    dist[1] = current_dist[1] - previous_dist[1]

    #calculate the average difference of all points
    movement = sum(dist)/len(dist)
    return movement

def extract_features(file_path):
    video_capture = cv2.VideoCapture(file_path)

    MUs = []
    previous_left_eye = []
    previous_right_eye = []
    previous_left_eyebrow = []
    previous_right_eyebrow = []
    previous_top_lip = []
    previous_bottom_lip = []
    previous_chin = []

    face_frame_number = 0

    while True:
        ret, frame = video_capture.read()

        #quit when the input video file ends
        if not ret:
            break

        #find all facial features in all the faces in the image
        face_landmarks_list = face_recognition.face_landmarks(frame)

        for face_landmarks in face_landmarks_list:

            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']
            left_eyebrow = face_landmarks['left_eyebrow']
            right_eyebrow = face_landmarks['right_eyebrow']
            top_lip = face_landmarks['top_lip']
            bottom_lip = face_landmarks['bottom_lip']
            chin = face_landmarks['chin']

            #calculate MUs only if data for a prevoius frame exists(not for first frame)
            if face_frame_number > 0:
                feature = numpy.zeros([12])

                #calculate the MUs
                feature[0] = movement_vertical_lip_center(previous_top_lip, top_lip)
                feature[1] = movement_vertical_lip_center(previous_bottom_lip, bottom_lip)
                feature[2], feature[3] = movement_left_lip_corner(previous_top_lip, previous_bottom_lip, top_lip, bottom_lip)
                feature[4], feature[5] = movement_right_lip_corner(previous_top_lip, previous_bottom_lip, top_lip, bottom_lip)
                feature[6] = movement_vertical_brow(previous_right_eyebrow, right_eyebrow)
                feature[7] = movement_vertical_brow(previous_left_eyebrow, left_eyebrow)
                feature[8], feature[9] = movement_vertical_cheeks(previous_chin, chin)
                feature[10] = eye_blinking(previous_right_eye, right_eye)
                feature[11] = eye_blinking(previous_left_eye, left_eye)

                MUs.append(feature)

            #save data from current frame to use it as previous data in the next step
            previous_left_eye = left_eye
            previous_right_eye = right_eye
            previous_left_eyebrow = left_eyebrow
            previous_right_eyebrow = right_eyebrow
            previous_top_lip = top_lip
            previous_bottom_lip = bottom_lip
            previous_chin = chin

            face_frame_number += 1

    video_capture.release()
    cv2.destroyAllWindows()

    return MUs
