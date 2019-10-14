import cv2
from ai_model import DogTypesModel

#  facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
model = DogTypesModel()


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        pred = model.predict(fr)
        print(pred)

        cv2.putText(fr, pred[0], (10, 10), font, 1, (255, 255, 0), 2)
        #  cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
