import cv2
import os

path = 'TrainDataset'

orb = cv2.ORB_create(nfeatures=1000)

# import images 
images = []
classNames = []
mylist = os.listdir(path)
print('Total classes Detected', len(mylist))

for cl in mylist:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findDes(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findID(img , desList , thres=15):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des,des2,k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass

    #print(matchList)

    if len(matchList)!=0:
        if max(matchList) > thres :
            finalVal = matchList.index(max(matchList))
    return finalVal




desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture("video.mp4")

while True:
    success, img2 = cap.read()

    if success:
        imgOriginal = img2.copy()

        img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

        id = findID(img2, desList)

        if id != -1:
            cv2.putText(imgOriginal,classNames[id],(30,60)
                        ,cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),4)

        cv2.imshow('img', imgOriginal)
        cv2.waitKey(1)


    else:
        break
