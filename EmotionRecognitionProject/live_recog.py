#================================================= VERSION 1 =====================================================
# import cv2
# import torch
# import torch.nn.functional as F
# from torchvision.transforms import ToTensor

# if __name__ == '__main__':
#     # Load your trained PyTorch model
#     model = torch.load("/Users/raagulsundar/EmotionRecognition/model_MK2")
#     model.eval()  # Set the model to evaluation mode

#     # Define emotion labels
#     emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

#     # Open webcam stream
#     cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam

#     while True:
#         ret, frame = cap.read()  # Read a frame from the webcam
#         if not ret:
#             break
        
#         # Preprocess the frame
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#         gray_frame = cv2.resize(gray_frame, (48, 48))  # Resize to match model's input size
#         tensor_frame = ToTensor()(gray_frame).unsqueeze(0)  # Convert to tensor and add batch dimension
#         tensor_frame = tensor_frame.float()  # Convert to float tensor

#         # Run inference
#         with torch.no_grad():
#             outputs = model(tensor_frame)
#             predicted_emotion = emotion_labels[outputs.argmax()]

#         # Overlay predicted emotion on the frame
#         cv2.putText(frame, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         # Display the frame
#         cv2.imshow('Emotion Recognition', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#             break

#     # Release the webcam and close the OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

#=================================================================================================================

#================================================= VERSION 2 =====================================================

# import cv2
# import torch
# from torchvision.transforms import ToTensor
# from emotion_model import EmotionModel  # Import your EmotionModel class here

# # Load your trained PyTorch model
# model = torch.load("/Users/raagulsundar/EmotionRecognition/model_MK2")
# model.eval()  # Set the model to evaluation mode

# # Define emotion labels
# emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# # Load a face detection cascade classifier (or use any other suitable method)
# face_cascade = cv2.CascadeClassifier("/Users/raagulsundar/EmotionRecognition/haarcascade_frontalface_default.xml")  # Replace with the actual path

# # Open webcam stream
# cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam

# while True:
#     ret, frame = cap.read()  # Read a frame from the webcam
#     if not ret:
#         break
    
#     # Convert frame to grayscale for face detection
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces using the cascade classifier
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     for (x, y, w, h) in faces:
#         face_roi = gray_frame[y:y+h, x:x+w]
#         face_tensor = ToTensor()(face_roi).unsqueeze(0).float()
        
#         # Run emotion recognition model
#         with torch.no_grad():
#             outputs = model(face_tensor)
#             predicted_emotion = emotion_labels[outputs.argmax()]
        
#         # Draw bounding box around the face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#         # Display predicted emotion inside the bounding box
#         cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow('Emotion Recognition', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#         break

# # Release the webcam and close the OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

#=================================================================================================================


#================================================= VERSION 3 =====================================================

import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from torchvision import transforms

# Load trained model
model = torch.load('/Users/raagulsundar/EmotionRecognition/model_MK2')
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MTCNN for face detection
mtcnn = MTCNN()

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use MTCNN for face detection
    faces = mtcnn.detect_faces(frame)

    for face_info in faces:
        x, y, w, h = face_info['box']
        face = gray[y:y+h, x:x+w]

        # Preprocess the face image
        face = cv2.resize(face, (48, 48))
        face = transforms.ToTensor()(face).unsqueeze(0)

        # Make emotion prediction
        with torch.no_grad():
            predictions = model(face)
        predicted_emotion = emotion_labels[predictions.argmax()]

        # Draw bounding box and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#=================================================================================================================

#================================================= VERSION 3 =====================================================
# import cv2
# import torch
# import dlib
# import numpy as np
# from torchvision import transforms

# # Load trained model
# model = torch.load('/Users/raagulsundar/EmotionRecognition/model_MK2')
# model.eval()

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Load emotion labels
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Initialize Dlib face detector
# detector = dlib.get_frontal_face_detector()

# while True:
#     ret, frame = cap.read()

#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Use Dlib for face detection
#     faces = detector(gray)

#     for face in faces:
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()
#         face_img = gray[y:y+h, x:x+w]

#         # Preprocess the face image
#         face_img = cv2.resize(face_img, (48, 48))
#         face_tensor = transforms.ToTensor()(face_img).unsqueeze(0)

#         # Make emotion prediction
#         with torch.no_grad():
#             predictions = model(face_tensor)
#         predicted_emotion = emotion_labels[predictions.argmax()]

#         # Draw bounding box and emotion label
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow('Facial Emotion Recognition', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#=================================================================================================================