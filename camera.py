import numpy as np
import sys
import cv2
from filter import apply_filter
from model import FaceKeypointsCaptureModel

rgb = cv2.VideoCapture('efgg.mp4')
length = int(rgb.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(rgb.get(cv2.CAP_PROP_FPS))
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels = {}

def __get_data__():
    """
    __get_data__: Gets data from the VideoCapture object and classifies them
    to a face or no face. 
    
    returns: tuple (faces in image, frame read, grayscale frame)
    """
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, fr, gray

def start_app(cnn):
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    width  = int(rgb.get(3)) # float
    height = int(rgb.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    PATH = 'something.webm'
    output = cv2.VideoWriter(PATH,fourcc, fps, (width,height))
    for sm in range(1,length-1):
        ix += 1
        faces, fr, gray_fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            roi = cv2.resize(fc, (96, 96))
            pred, pred_dict = cnn.predict_points(roi[np.newaxis, :, :, np.newaxis])
            pred, pred_dict = cnn.scale_prediction((x, fc.shape[1]+x), (y, fc.shape[0]+y))

            fr = apply_filter(fr, pred_dict)
        sys.stdout.write(f"writing...{int((sm/length)*100)+1}%\n")
        sys.stdout.flush()
        output.write(fr)
        if cv2.waitKey(1) == 27:
            break
       # cv2.imshow('Filter', fr)
    rgb.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    model = FaceKeypointsCaptureModel("face_model.json", "face_model.h5")
    start_app(model)