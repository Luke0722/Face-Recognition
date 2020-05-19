import cv2
import imutils

# 放入cascade.xml資源檔的路徑
casc_path = r'.\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
# 載入分類器
face_cascade = cv2.CascadeClassifier(casc_path)
# 指定要使用那一隻攝影機（0 代表第一隻、1 代表第二隻）。
cap = cv2.VideoCapture(0)

while True:
    # 從視訊鏡頭擷取畫面
    # cap.read() 回傳值：
    #     1.ret：成功與否（True 代表成功，False 代表失敗）
    #     2.frame：鏡頭的單張畫面
    reg, frame = cap.read()
    # 縮小圖片
    frame = imutils.resize(frame, width=800)
    # 轉換成灰階圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 檢測人臉，注意要轉換成灰階圖
    # detectMultiScale 參數:
    #     1.圖片數據
    #     2.scaleFactor 每次搜尋方塊減少的比例
    #     3.minNeighbors 每個目標至少檢測到幾次以上，才可被認定是真數據。
    #     4.minSize 設定數據搜尋的最小尺寸 ，如 minSize=(40,40)
    face_rects = face_cascade.detectMultiScale(gray, 1.08, 3)

    #偵測出來的結果的資料結構(x, y, w, h)
    for (x,y,w,h) in face_rects:
        # 透過OpenCV來把邊界框畫出來
        # rectangle 參數:
        #     1.要畫矩形的圖片
        #     2.左上角座標 tuple (x, y)
        #     3.右下角座標 tuple (x, y)
        #     4.邊框顏色 tuple (r,g,b)
        #     5.邊框寬度 int
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顯示影片截圖(視窗名字,圖片)
    cv2.imshow('Face Detector', frame)
    # 按q結束程式
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
cap.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
