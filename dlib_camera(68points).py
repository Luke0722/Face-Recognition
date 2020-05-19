import dlib
import cv2

# 指定要使用那一隻攝影機（0 代表第一隻、1 代表第二隻）
cap = cv2.VideoCapture(0)
# 取得dlib預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
# 根據shape_predictor方法載入68個特徵點模型
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 以無窮迴圈從影片檔案讀取畫面，cap.isOpened()回傳true代表鏡頭有正確開啟
while(cap.isOpened()):
  # 從視訊鏡頭擷取畫面
  ret, frame = cap.read()
  # 偵測人臉
  face_rects, scores, idx = detector.run(frame, 0)
  # 取出所有偵測的結果
  for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    # 以方框標示人臉
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
    # 標示分數與子偵測器
    text = "%2.2f(%d)" % (scores[i], idx[i])
    cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    # 標示68特徵點
    shape = predictor(frame, d)
    for (index, point) in enumerate(shape.parts()):
        cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), 1)

  # 顯示結果
  cv2.imshow("Face Detection", frame)
  # 按q跳出迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()