import dlib
import cv2
import imutils

# 取得Dlib預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
# 根據shape_predictor方法載入68個特徵點模型
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 讀取照片圖檔
img = cv2.imread('photo.jpg')
# 縮小照片
img = imutils.resize(img, width=700)
# 偵測人臉，輸出分數
face_rects, scores, idx = detector.run(img, 0, -1)
# 偵測人臉
face_rects = detector(img, 0)

# 取出所有偵測的結果
for (index, face) in enumerate(face_rects):
  x1 = face.left()
  y1 = face.top()
  x2 = face.right()
  y2 = face.bottom()
  # 標示偵測的人臉
  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
  # 標示分數與子偵測器
  text = "%2.2f(%d)" % (scores[index], idx[index])
  cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
  # 標示特徵點
  shape = predictor(img, face)
  for(index,point) in enumerate(shape.parts()):
    cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), 1)

# 顯示結果
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

