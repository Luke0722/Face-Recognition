import cv2
import imutils

# 欲偵測圖片的路徑
image_path = 'photo.jpg'
# 放入cascade.xml資源檔的路徑
casc_path = r'.\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
# 載入分類器
face_cascade = cv2.CascadeClassifier(casc_path)
# 讀取圖片
image = cv2.imread(image_path)

# 縮小照片
image = imutils.resize(image, width=300)
# 轉成灰階圖片
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 檢測人臉，注意要轉換成灰階圖
# detectMultiScale 參數:
#     1.圖片數據
#     2.scaleFactor 每次搜尋方塊減少的比例
#     3.minNeighbors 每個目標至少檢測到幾次以上，才可被認定是真數據。
#     4.minSize 設定數據搜尋的最小尺寸 ，如 minSize=(40,40)
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.08,minNeighbors=3)

print('Found {0} faces!'.format(len(faces)))

# 用方框標記偵測到的人臉
for (x, y, w, h) in faces:
    # rectangle 參數:
    #     1.要畫矩形的圖片
    #     2.左上角座標 tuple (x, y)
    #     3.右下角座標 tuple (x, y)
    #     4.邊框顏色 tuple (r,g,b)
    #     5.邊框寬度 int
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 顯示圖片(視窗名字,圖片)
cv2.imshow("Faces found", image)
# 儲存圖片
# cv2.imwrite("face_detection.jpg", image)
# 等待按下任一按鍵
cv2.waitKey(0)
# 關閉顯示圖片的視窗
cv2.destroyAllWindows()
