import os
import pickle
import numpy
from hmmlearn import hmm
from feature_extraction import extract_features

dataset_path = '../ExpressionRecognition-main/ExpressionRecognition-main/emotion recognition/trainingData'

happy_features = []
angry_features = []
surprise_features = []
disgust_features = []
fear_features = []
sad_features = []
sequence_features = []

happy_prediction = []
angry_prediction = []
surprise_prediction = []
disgust_prediction = []
fear_prediction = []
sad_prediction = []

emotion_states = ["Happy", "Angry", "Surprise", "Disgust", "Fear", "Sad", "Neutral"]

print("extracting features from files")
for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

    #iterate through all subfolders
    if dirpath is not dataset_path:
        dirpath_parts = dirpath.split("\\")
        emotion = dirpath_parts[-1]
        print(emotion)

        #sort the data into the classes based on the folder name
        for file in filenames:
            print(file)
            features = extract_features(dataset_path + "/" + emotion + "/" + file)
            if emotion == "Happy":
                happy_features.extend(features)
            elif emotion == "Angry":
                angry_features.extend(features)
            elif emotion == "Surprise":
                surprise_features.extend(features)
            elif emotion == "Disgust":
                disgust_features.extend(features)
            elif emotion == "Fear":
                fear_features.extend(features)
            elif emotion == "Sad":
                sad_features.extend(features)
            elif emotion == "Sequence":
                sequence_features.extend(features)

print("features extracted")

happy_model = hmm.GaussianHMM(n_components=10, covariance_type="full")
angry_model = hmm.GaussianHMM(n_components=10, covariance_type="full")
surprise_model = hmm.GaussianHMM(n_components=10, covariance_type="full")
disgust_model = hmm.GaussianHMM(n_components=10, covariance_type="full")
fear_model = hmm.GaussianHMM(n_components=10, covariance_type="full")
sad_model = hmm.GaussianHMM(n_components=10, covariance_type="full")

happy_model.fit(happy_features)
angry_model.fit(angry_features)
surprise_model.fit(surprise_features)
disgust_model.fit(disgust_features)
fear_model.fit(fear_features)
sad_model.fit(sad_features)

pickle.dump(happy_model, open("../ExpressionRecognition-main/ExpressionRecognition-main/models/happy_model.pkl", "wb"))
pickle.dump(angry_model, open("../ExpressionRecognition-main/ExpressionRecognition-main/models/angry_model.pkl", "wb"))
pickle.dump(surprise_model, open("../ExpressionRecognition-main/ExpressionRecognition-main/models/surprise_model.pkl", "wb"))
pickle.dump(disgust_model, open("../ExpressionRecognition-main/ExpressionRecognition-main/models/disgust_model.pkl", "wb"))
pickle.dump(fear_model, open("../ExpressionRecognition-main/ExpressionRecognition-main/models/fear_model.pkl", "wb"))
pickle.dump(sad_model, open("../ExpressionRecognition-main/ExpressionRecognition-main/models/sad_model.pkl", "wb"))

print("models fitted and saved")

happy_prediction = happy_model.predict(sequence_features)
angry_prediction = angry_model.predict(sequence_features)
surprise_prediction = surprise_model.predict(sequence_features)
disgust_prediction = disgust_model.predict(sequence_features)
fear_prediction = fear_model.predict(sequence_features)
sad_prediction = sad_model.predict(sequence_features)

emotion_features = []

#save the feature vector for every frame containing the predictions of the 6 emotion specific HMMs
for i in range(len(happy_prediction)):
    emotion_features.append([happy_prediction[i], angry_prediction[i], surprise_prediction[i], disgust_prediction[i], fear_prediction[i], sad_prediction[i]])
  
print("feature vector for high-level HMM generated")

emotion_model = hmm.GaussianHMM(n_components=7, covariance_type="full",
                  init_params="cm", params="cmt")
#set start state and transition matrix
emotion_model.startprob_ = numpy.array([0, 0, 0, 0, 0, 0, 1])
emotion_model.transmat_ = numpy.array([[0.5, 0, 0, 0, 0, 0, 0.5],
                          [0, 0.5, 0, 0, 0, 0, 0.5],
                          [0, 0, 0.5, 0, 0, 0, 0.5],
                          [0, 0, 0, 0.5, 0, 0, 0.5],
                          [0, 0, 0, 0, 0.5, 0, 0.5],
                          [0, 0, 0, 0, 0, 0.5, 0.5],
                        [(1/7), (1/7), (1/7), (1/7), (1/7), (1/7), (1/7)]])
emotion_model.fit(emotion_features)

pickle.dump(emotion_model, open("../ExpressionRecognition-main/ExpressionRecognition-main/models/emotion_model.pkl", "wb"))

print("emotion-model fitted and saved")
