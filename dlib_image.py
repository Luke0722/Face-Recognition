import dlib
import cv2
import imutils

# 讀取照片圖檔
img = cv2.imread('photo.jpg')
# 縮小照片
img = imutils.resize(img, width=700)
# 取得dlib預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
# 偵測人臉
# detector
# 參數：
#     1.圖檔
#     2.反取樣（unsample）的次數：如果圖片太小的時候，將其設為 1 可讓程式偵較容易測出更多的人臉
# 回傳：
#     1.人臉的位置：臉孔的（x,y,w,h）為（d.left(), d.top(), d.right, d.bottom()）
face_rects = detector(img, 0)
# 取出所有偵測的結果
for i, d in enumerate(face_rects):
  x1 = d.left()
  y1 = d.top()
  x2 = d.right()
  y2 = d.bottom()
  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

# 顯示結果
cv2.imshow("Face Detection", img)
# 等待按下任一按鍵
cv2.waitKey(0)
# 關閉顯示圖片的視窗
cv2.destroyAllWindows()
