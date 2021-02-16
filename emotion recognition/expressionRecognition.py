import pickle
import face_recognition
import numpy
from cv2 import cv2
from PIL import Image, ImageDraw, ImageFont
from feature_extraction import movement_vertical_lip_center, movement_left_lip_corner, movement_right_lip_corner, movement_vertical_brow, movement_vertical_cheeks, eye_blinking

font_size = 75
font = ImageFont.truetype("arial.ttf", font_size)

features = []
emotions = []
emotion_states = ["Happy", "Angry", "Surprise", "Disgust", "Fear", "Sad", "Neutral"]

#load trained models
happy_model = pickle.load(open("../ExpressionRecognition-main/ExpressionRecognition-main/emotion recognition/models/happy_model.pkl", "rb"))
angry_model = pickle.load(open("../ExpressionRecognition-main/ExpressionRecognition-main/emotion recognition/models/angry_model.pkl", "rb"))
surprise_model = pickle.load(open("../ExpressionRecognition-main/ExpressionRecognition-main/emotion recognition/models/surprise_model.pkl", "rb"))
disgust_model = pickle.load(open("../ExpressionRecognition-main/ExpressionRecognition-main/emotion recognition/models/disgust_model.pkl", "rb"))
fear_model = pickle.load(open("../ExpressionRecognition-main/ExpressionRecognition-main/emotion recognition/models/fear_model.pkl", "rb"))
sad_model = pickle.load(open("../ExpressionRecognition-main/ExpressionRecognition-main/emotion recognition/models/sad_model.pkl", "rb"))
emotion_model = pickle.load(open("../ExpressionRecognition-main/ExpressionRecognition-main/emotion recognition/models/emotion_model.pkl", "rb"))

def predict_emotion():
    happy_prediction = happy_model.predict(features)
    angry_prediction = angry_model.predict(features)
    surprise_prediction = surprise_model.predict(features)
    disgust_prediction = disgust_model.predict(features)
    fear_prediction = fear_model.predict(features)
    sad_prediction = sad_model.predict(features)

    #save the feature vector for every frame containing the predictions of the 6 emotion specific HMMs
    emotion_features = []
    for i in range(len(happy_prediction)):
        emotion_features.append([happy_prediction[i], angry_prediction[i], surprise_prediction[i], disgust_prediction[i], fear_prediction[i], sad_prediction[i]])

    #get the prediction of the current frame
    emotion_prediction = emotion_model.predict(emotion_features)[len(emotion_features)-1]
    return emotion_states[emotion_prediction]


def expression_recognition(filename):

    face_frame_number = 0

    previous_left_eye = []
    previous_right_eye = []
    previous_left_eyebrow = []
    previous_right_eyebrow = []
    previous_top_lip = []
    previous_bottom_lip = []
    previous_chin = []

    video_capture = cv2.VideoCapture(filename)
    if video_capture.isOpened() is False:
        return -1


    while True:
        ret, frame = video_capture.read()

        #quit when the input video file ends
        if not ret:
            break

        # find all facial features in all the faces in the image
        face_landmarks_list = face_recognition.face_landmarks(frame)

        pil_image = Image.fromarray(frame)
        width_img, high_img = pil_image.size
        draw = ImageDraw.Draw(pil_image)

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
                MUs = numpy.zeros([12], dtype=float)

                #calculate the MUs
                MUs[0] = movement_vertical_lip_center(previous_top_lip, top_lip)
                MUs[1] = movement_vertical_lip_center(previous_bottom_lip, bottom_lip)
                MUs[2], MUs[3] = movement_left_lip_corner(previous_top_lip, previous_bottom_lip, top_lip, bottom_lip)
                MUs[4], MUs[5] = movement_right_lip_corner(previous_top_lip, previous_bottom_lip, top_lip, bottom_lip)
                MUs[6] = movement_vertical_brow(previous_right_eyebrow, right_eyebrow)
                MUs[7] = movement_vertical_brow(previous_left_eyebrow, left_eyebrow)
                MUs[8], MUs[9] = movement_vertical_cheeks(previous_chin, chin)
                MUs[10] = eye_blinking(previous_right_eye, right_eye)
                MUs[11] = eye_blinking(previous_left_eye, left_eye)

                #calculate the emotion based on all previous extracted features
                features.append(MUs)
                emotions.append(predict_emotion())

                #add the emotion as text to the top middel of the image only when more than 3 frames classified as the same emotion
                if face_frame_number-1 >=2 and emotions[face_frame_number-1] == emotions[face_frame_number-2] == emotions[face_frame_number-3]:
                    width_txt, high_txt = draw.textsize(emotions[face_frame_number-1], font)
                    draw.text(((width_img - width_txt)/2, 0), emotions[face_frame_number-1], font=font)

            face_frame_number += 1

            #save data from current frame to use it as previous data in the next step
            previous_left_eye = left_eye
            previous_right_eye = right_eye
            previous_left_eyebrow = left_eyebrow
            previous_right_eyebrow = right_eyebrow
            previous_top_lip = top_lip
            previous_bottom_lip = bottom_lip
            previous_chin = chin

        cv2.imshow('Video', numpy.uint8(pil_image))

        # quit loop by either hitting 'q' or pressing the close button
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break

    video_capture.release()
    cv2.destroyAllWindows()
