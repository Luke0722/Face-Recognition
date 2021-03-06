import sys, os, dlib, glob
import cv2
import imutils
import numpy as np
from skimage import io
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

# get dlib face detector
detector = dlib.get_frontal_face_detector()
# get 68 face landmarks predictor
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# get face aligner 
fa = FaceAligner(predictor, desiredFaceWidth=256)
# get face  embedding model 
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# assign the camera (number 0)
cap = cv2.VideoCapture(0)
# the route of the database folder which contains face photos 
faces_folder_path = "./rec"
# the candidate list (filenames of face photos)
candidate = []
# the candidates' face feature list
descriptors = []

# read the photos in the database folder, put candidates' name into cadidate array, and put each face's feature vector into discription array
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    # get the filename(contain filename extension)
    base = os.path.basename(f)
    # get the filename(no filename extension) and append to candidate array
    candidate.append(os.path.splitext(base)[0])
    # put file image into variable img
    img = io.imread(f)
    # get grayscale frame (for the face aligner method)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 1. Face Detection
    face_rects = detector(img, 0)
    for index, face in enumerate(face_rects):
        # 2. Face Alignment
        faceAligned = fa.align(img, gray, face)
        # 3. Re Face Detection
        face_rects2 = detector(faceAligned, 1)
        for index2, face2 in enumerate(face_rects2):
            # 4. 68 Face Landmarks Detection
            shape = predictor(faceAligned, face2)
            # 5. 128D Face Feature Vector Embedding
            face_descriptor = facerec.compute_face_descriptor(faceAligned, shape)
            # put face feature vector into numpy array 
            v = np.array(face_descriptor)
            # append the face feature vector into description array
            descriptors.append(v)

# use loop to fetch camera screen and show the result
while(cap.isOpened()):
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    # fetch the camera screen
    ret, frame = cap.read()
    # adjust frame size
    frame = imutils.resize(frame, width=800)
    # convert frame from BGR to RGB (generally color channel)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # get grayscale frame (for the face aligner method)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 1. Face Detection
    face_rects= detector(frame, 1)
    for index, rect in enumerate(face_rects):
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        # Mark the face using rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
        # 2. Face Alignment
        faceAligned= fa.align(frame, gray, rect)
        # 3. Re Face Detection
        face_rects2 = detector(faceAligned, 1)
        for index2, rect2 in enumerate(face_rects2):
            # 4. 68 Face Landmarks Detection
            shape = predictor(faceAligned, rect2)
            # 5. 128D Face Feature Vector Embedding
            face_descriptor = facerec.compute_face_descriptor(faceAligned, shape)
            # put face feature vector into numpy array 
            d_test = np.array(face_descriptor)
            #  calcualte euclidean distance of the testing person and each candiate
            dist = [] 	# intialize
            for index in descriptors:
                # calcualte euclidean distance
                dist_ = np.linalg.norm(index - d_test)
                # put euclidean distance into dist list
                dist.append(dist_)
            # Recognize
            if(dist!=[]):
                # match each candidate's name and euclidean distance into a dictionary
                c_d = dict(zip(candidate, dist))
                # descending sort the dictionary [("name",distance)] according to euclidean distance
                cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
                # put the candidate's name who has the smallet distance intorec_name, 0.5 is the threshold(adjustable)
                if (cd_sorted[0][1] < 0.5):
                    rec_name = cd_sorted[0][0]
                else:
                    rec_name = "No Data"

            # tag the name of recognized person (Chinese)
            imgPil = Image.fromarray(frame)
            font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 20)
            draw = ImageDraw.Draw(imgPil)
            draw.fontmode = '1' 
            draw.text((x1, y1-20), rec_name,font=font, fill=(255,255,255))
            frame = np.array(imgPil)

            # tag the name of recognized person (English only)
            # cv2.putText(frame, rec_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # convert the frame from RGB to BGR (for OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # show the result
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()