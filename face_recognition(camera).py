import sys, os, dlib, glob
from skimage import io
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

# 取得dlib預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
# 根據shape_predictor方法載入68個特徵點模型
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 指定要使用那一隻攝影機（0 代表第一隻、1 代表第二隻）
cap = cv2.VideoCapture(0)
# 比對人臉圖片的資料夾名稱
faces_folder_path = "./rec"
# 比對人臉圖片的描述子列表
descriptors = []
# 比對人臉圖片的名稱列表
candidate = []

# 讀取資料夾裡的圖片，將人名(圖檔名)存入candidate陣列，將每張圖的128維特徵向量存入description陣列
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    # 取得圖片檔名(含副檔名)
    base = os.path.basename(f)
    # 取得圖片檔名(不含副檔名)，並存到candidate一維陣列中
    candidate.append(os.path.splitext(base)[0])
    # 取得圖片放到img變數
    img = io.imread(f)
    # 1.人臉偵測
    face_rects = detector(img, 1)

    for index, face in enumerate(face_rects):
        # 2.68特徵點偵測
        shape = predictor(img, face)
        # 3.128維特徵向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        # 將特徵向量轉換成numpy array格式
        v = np.array(face_descriptor)
        # 將數據加入到描述子列表description一維陣列中
        descriptors.append(v)

# 以迴圈從影片檔案讀取擷取畫面
while(cap.isOpened()):
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    # 從視訊鏡頭擷取畫面
    ret, frame = cap.read() 
    # 縮小圖片
    frame = imutils.resize(frame, width=800)
    # 1.人臉偵測
    face_rects, scores, idx = detector.run(frame, 0)
    # 取出所有偵測的結果(所有人臉座標點)
    for index, face in enumerate(face_rects):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # 以方框標示偵測的人臉
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
        # 2.68特徵點偵測
        shape = predictor(frame, face)
        # 3.128維特徵向量
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        # 轉換numpy array格式
        d_test = np.array(face_descriptor)

        # 4.計算歐式距離  (資料夾有幾張照片就會得到幾個距離)
        # 存放此張照片與資料夾人臉距離的陣列(每次迴圈都清空)
        dist = []    
        for index in descriptors:
            # 計算距離
            dist_ = np.linalg.norm(index - d_test)
            # 將距離加入陣列
            dist.append(dist_)

        # 5.辨識人名
        if(dist!=[]):
            # 將人名與歐式距離組成一個字典(dictionary)
            c_d = dict(zip(candidate, dist))
            # 根據歐式距離由小到大排序 [("名字",距離)]二維陣列
            cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
            # 將辨識結果存入rec_name，設定0.5作為最低辨識標準(歐式距離(0~1)越小越像)
            if (cd_sorted[0][1] < 0.5):
                rec_name = cd_sorted[0][0]
            else:
                rec_name = "No Data"
        # 標示辨識的人名
        cv2.putText(frame, rec_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

     # 顯示結果
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()