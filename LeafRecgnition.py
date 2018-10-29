import numpy as numpy
import cv2

m_x=0
m_y=0
m_flag=False

#################################################
##
#################################################
def detect_markerPt(match_pt):
    x1,y1=0,0
    x2,y2=0,0
    cnt1=0
    cnt2=0
    for i in range(len(match_pt)):
        nx = match_pt[i].pt[0]
        ny = match_pt[i].pt[1]
        if i > 0:
            p1 = numpy.array([x1,y1])
            p2 = numpy.array([nx,ny])
            dis  = p2 - p1
            n2norm = numpy.linalg.norm(dis)
            if n2norm < 300:
                cnt1+=1
                x1 = (x1 + nx)/cnt1
                y1 = (y1 + ny)/cnt1
            else:
                cnt2+=1
                x2 = (x2 + nx)/cnt2
                y2 = (y2 + ny)/cnt2
        else:
            x1 += nx
            y1 += ny
            cnt1+=1

    return (int(x1),int(y1)),(int(x2),int(y2))

#################################################
##
#################################################
def mouse_event(event, x, y, flags, param):
    global m_x
    global m_y
    global m_flag

    if event == cv2.EVENT_LBUTTONUP:
	# 左クリック押下
        m_x=x
        m_y=y

        m_flag=True


cap = cv2.VideoCapture('20180925.MP4')
#cap = cv2.VideoCapture(0)

# 特徴抽出器の生成
detector = cv2.xfeatures2d.SIFT_create()
# テンプレート画像
temp = cv2.imread('template4.png')
# temp_resize = cv2.resize(temp,(int(temp.shape[1]/4),int(temp.shape[0]/4)))
#temp_resize = cv2.imread('template2.png')

cv2.namedWindow('Raw Frame')
cv2.setMouseCallback('Raw Frame',mouse_event)

while(cap.isOpened()):
    ret1, frame = cap.read()
    # ret1, frame_resize = cap.read()

    # # グレースケール変換
    # frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # # 画像しきい値を判別しやすくするためにガウシアンブラーをかける
    # frame_blur = cv2.GaussianBlur(frame_gray,(11,11),0)

    # # 画像全体を2値化
    # # 大津アルゴを採用
    # ret2, th1 = cv2.threshold(frame_blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # # ピクセル値そのものでなく周囲のピクセルとの関係も考慮にいれた2値化
    # # 11ピクセルの範囲
    # # 11ピクセルの範囲から計算した値から3を引いたものが最終的な閾値
    # # ADAPTIVE_THRESH_GAUSSIAN_C：範囲内のピクセルの平均をとる
    # th2 = cv2.adaptiveThreshold(frame_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)


    # # 画像をリサイズ
    # frame_rslt = cv2.resize(fraqme_blur,(int(frame.shape[1]/4),int(frame.shape[0]/4)))
    # frame_rslt1 = cv2.resize(th1,(int(frame.shape[1]/4),int(frame.shape[0]/4)))
    # frame_rslt2 = cv2.resize(th2,(int(frame.shape[1]/4),int(frame.shape[0]/4)))

    frame_resize = cv2.resize(frame,(int(frame.shape[1]/4),int(frame.shape[0]/4)))
    #マッチングテンプレートを実行
    #比較方法はTM_CCOEFF_NORMED
    result = cv2.matchTemplate(frame_resize, temp, cv2.TM_CCOEFF_NORMED)
    #kpは特徴点の位置、desは特徴を表すベクトル
    kp1, des1 = detector.detectAndCompute(frame_resize, None)
    kp2, des2 = detector.detectAndCompute(temp, None)

    #特徴点の比較器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    #割合試験を適用
    good = []
    good_queryPt = []
    match_param = 0.6
    for m,n in matches:
        if m.distance < match_param * n.distance:
            good.append([m])
            good_queryPt.append(kp1[m.queryIdx])

    result_img = cv2.drawMatchesKnn(frame_resize,kp1,temp,kp2,good,None,flags=2)
    # frame_rslt = cv2.resize(result_img,(int(result_img.shape[1]/4),int(result_img.shape[0]/4)))
    # cv2.imshow('rslt',frame_rslt)

    # print(good_queryPt[0].pt[0]) x座標

    if len(good_queryPt)>0:
        left_mrk,right_mrk = detect_markerPt(good_queryPt)
        if ( left_mrk!=(0,0) ) and ( right_mrk!=(0,0) ) :
            cv2.rectangle(result_img,left_mrk,right_mrk,(255,0,0))

    cv2.imshow('Raw Frame',result_img)

    if m_flag:
        m_flag=False
        hsv = cv2.cvtColor(result_img,cv2.COLOR_BGR2HSV_FULL)
        h = hsv[m_y,m_x,0]
        s = hsv[m_y,m_x,1]
        print(h,s)


    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('a'):
        cv2.imwrite('capture.jpg',frame)
        break

cap.release()
cv2.destroyAllWindows()