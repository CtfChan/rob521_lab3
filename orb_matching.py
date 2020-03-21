import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

filename1 = './l3_mapping_data/camera_image_1.jpeg'
filename2 = './l3_mapping_data/camera_image_2.jpeg'


img1 = cv.imread(filename1, cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread(filename2, cv.IMREAD_GRAYSCALE) # trainImage

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

print("kp1")
print(kp1[0].pt)



# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
print("length of matches: ", len(matches))
print(dir(matches[0]))
print(matches[0].imgIdx)
print(matches[0].queryIdx)
print(matches[0].trainIdx)


# Draw all matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:len(matches)],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()