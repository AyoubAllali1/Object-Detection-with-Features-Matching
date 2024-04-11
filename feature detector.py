import cv2

# 0 pour grey
img1 = cv2.imread('TrainDataset/SimoLife.png',0)
img2 = cv2.imread('TestDataset/image 2.png',0)

## nfeatures optional
orb = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

#print(des1[0])

imgkp1 = cv2.drawKeypoints(img1,kp1,None)
imgkp2 = cv2.drawKeypoints(img2,kp2,None)

#Matching 

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2,k=2)

good = []

for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print(good)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)



cv2.imshow('kp1',imgkp1)
cv2.waitKey(0)
cv2.imshow('kp2',imgkp2)
cv2.waitKey(0)

cv2.imshow('img1',img1)
cv2.waitKey(0)
cv2.imshow('img2',img2)
cv2.waitKey(0)
cv2.imshow('img3',img3)
cv2.waitKey(0)







