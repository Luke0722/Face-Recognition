# Real Time Face Recognition 即時人臉辨識

This project use OpenCV and pre-trained models of Dlib、Imutils to execute real time face recognition.

Based on one-shot learning concept, it only needs one face photo per person for recognizing.

此專案利用Dlib和Imutils的預訓練模型(pre-trained model)，進行即時人臉識別。

基於one-shot learning的概念，這個模型只要一張照片就可以辨識出對象。

## How to execute (執行方式)
#### 1.New a folder named "rec" and put face photo in it. 新增一個rec的資料夾，將欲識別的人臉照片放在資料夾中

#### 2.Adjust the route 調整路徑：

    # The route of face photo folder  圖片資料夾(rec)路徑
    faces_folder_path = "./rec"
    
    # The route of Model 模型檔路徑   
    predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

#### 3. Execute 直接執行

## Model(模型檔)
#### 1. Face Detection 人臉偵測 
`detector = dlib.get_frontal_face_detector()`

From Dlib：http://dlib.net/face_detector.py.html

使用HOG特徵提取與SVM分類器，Dataset為LFW(2825 images)

#### 2. 68 face_landmarks 68人臉特徵點檢測
`predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")`

From Dlib：https://github.com/davisking/dlib-models

使用HOG特徵提取，Dataset為ibug 300-W

#### 3. Face Alignment 人臉校正 
`fa = FaceAligner(predictor, desiredFaceWidth=256)`

From Imutils：https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

#### 4. 128 Dimension Face Vector (Face Encoding) 擷取128維人臉特徵向量 
`facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")`

From Dlib：https://github.com/davisking/dlib-models

模型框架為34層的resnet架構(從facenet 34層resnet架構修改而來)，Dataset包括face scrub dataset、VGG dataset 等超過3百萬張人臉照片

## Demo images(執行畫面)
<img width="600"  src="screenshots/Demo Image.jpg"/>

## 參考資料
##### 基於python語言使用OpenCV搭配dlib實作人臉偵測與辨識 By 張鈞名 https://www.tpisoftware.com/tpu/articleDetails/950
##### Face Landmark & Alignment By CH.Tseng https://chtseng.wordpress.com/2018/08/18/face-landmark-alignment/
