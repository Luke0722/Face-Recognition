import sys, os, dlib, glob
import cv2
import imutils
import numpy as np
from skimage import io
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

# 取得dlib預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
# 根據shape_predictor方法載入68個特徵點模型
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 載入imutils的人臉校正器
fa = FaceAligner(predictor, desiredFaceWidth=256)
# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 指定要使用那一隻攝影機（0 代表第一隻、1 代表第二隻）
cap = cv2.VideoCapture(0)
# 比對人臉圖片資料夾名稱
faces_folder_path = "./rec"
# 比對人臉名稱列表
candidate = []
# 比對人臉描述子列表
descriptors = []

# 讀取資料夾裡的圖片，將人名(圖檔名)存入candidate陣列，將每張圖的128維特徵向量存入description陣列
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    # 取得圖片檔名(含副檔名)
    base = os.path.basename(f)
    # 取得圖片檔名(不含副檔名)，並存到candidate一維陣列中
    candidate.append(os.path.splitext(base)[0])
    # 取得圖片放到img變數
    img = io.imread(f)
    # 取得灰階圖(人臉校正使用)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 1.人臉偵測
    face_rects = detector(img, 0)
    for index, face in enumerate(face_rects):
        # 2.人臉校正
        faceAligned = fa.align(img, gray, face)
        # 3.再人臉偵測
        face_rects2 = detector(faceAligned, 1)
        for index2, face2 in enumerate(face_rects2):
            # 4.68特徵點偵測
            shape = predictor(faceAligned, face2)
            # 5.128維特徵向量
            face_descriptor = facerec.compute_face_descriptor(faceAligned, shape)
            # 將特徵向量轉換成numpy array格式
            v = np.array(face_descriptor)
            # 將數據加入到描述子列表description一維陣列中
            descriptors.append(v)

# 以迴圈從影片檔案讀取影格，並顯示出來
while(cap.isOpened()):
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    # 從視訊鏡頭擷取畫面
    ret, frame = cap.read()
    # 縮小圖片
    frame = imutils.resize(frame, width=800)
    # 由OpenCV的BGR影像轉換成RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 取得灰階圖
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 1.人臉偵測
    face_rects= detector(frame, 1)
    # 取出所有偵測的結果(所有人臉座標點)
    for index, rect in enumerate(face_rects):
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        # 以方框標示偵測的人臉
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
        # 2.人臉校正
        faceAligned= fa.align(frame, gray, rect)
        # 3.再人臉偵測
        face_rects2 = detector(faceAligned, 1)
        for index2, rect2 in enumerate(face_rects2):
            # 4.68特徵點偵測
            shape = predictor(faceAligned, rect2)
            # 5.128維特徵向量
            face_descriptor = facerec.compute_face_descriptor(faceAligned, shape)
            # 將特徵向量轉換成numpy array格式
            d_test = np.array(face_descriptor)
            # 計算歐式距離   (資料夾有幾張照片就會得到幾個距離)
            dist = [] 	# 初始化 存放人臉距離的陣列
            for index in descriptors:
                # 計算距離
                dist_ = np.linalg.norm(index - d_test)
                # 將距離加入陣列
                dist.append(dist_)
            # 辨識人名
            if(dist!=[]):
                # 將人名與歐式距離組成一個字典(dictionary)
                c_d = dict(zip(candidate, dist))
                # 根據歐式距離由小到大排序 [("名字",距離)]二微陣列
                cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
                # 將辨識結果存入rec_name，設定0.5作為閥值(歐式距離(0~1)越小越像)
                if (cd_sorted[0][1] < 0.5):
                    rec_name = cd_sorted[0][0]
                else:
                    rec_name = "No Data"

            # 標示辨識的人名(中文)
            imgPil = Image.fromarray(frame)
            font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 20)
            draw = ImageDraw.Draw(imgPil)
            draw.fontmode = '1' # 關閉反鋸齒
            draw.text((x1, y1-20), rec_name,font=font, fill=(255,255,255))
            frame = np.array(imgPil)

            # 標示辨識的人名(只能標示英文)
            # cv2.putText(frame, rec_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #將圖片轉為OpenCV使用的BGR圖像
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 顯示結果
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()