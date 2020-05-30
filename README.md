# Real time face-Recognition
Real Time Face Recognition using Opencv and Dlib in Python.

## 即時臉部辨識
#### 程式碼：[face_recognition(camera).py](face_recognition(camera).py)
#### 需要的模型檔：[shape_predictor_68_face_landmarks.dat](shape_predictor_68_face_landmarks.dat)、[dlib_face_recognition_resnet_model_v1.dat](dlib_face_recognition_resnet_model_v1.dat)
#### 使用套件：OpenCV、Dlib、Imutils、Scipy、Skimage、Pillow
#### 執行方式：
##### 1.新增一個rec的資料夾，將人臉照片放在資料夾中

##### 2.調整路徑：

    #rec資料夾路徑
    faces_folder_path = "./rec"
    
    #模型檔路徑   
    predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

##### 3.直接執行

## 參考資料
基於python語言使用OpenCV搭配dlib實作人臉偵測與辨識
https://tpu.thinkpower.com.tw/tpu/articleDetails/950
