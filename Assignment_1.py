
# coding: utf-8

# In[2]:

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import scipy.ndimage as ndi
import scipy as sp
import skimage
import math
import cv2


get_ipython().magic('matplotlib inline')


# # Question 1

# In[3]:

A = plt.imread("greenscreen.jpg")

# make an array of the image
im = np.array(A)
# create new array for background
B = plt.imread("background.jpg")
backg = np.array(B)

plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(A)
plt.show()


# In[4]:

# fill output array with zeros
width = A.shape[0]
height = A.shape[1]

BW = np.zeros((width, height, 3), dtype = "uint8")

# seperate image into color channels
r = im[:, :, 0]
g = im[:, :, 1]
b = im[:, :, 2]
'''
#test for green pixel
if g > 50 and b > 50 and r < 10:
    im[i, j] = im2[i, j]
'''

R = r < 10
G = g > 100
B = b > 50

BW = np.logical_and(R, G, B)

'''
#display new image

plt.imshow(im)
plt.show()

plt.imshow(im2)
plt.show()
'''
plt.figure(figsize=(10, 10), dpi=100)
#BW = cv2.cvtColor(BW, cv2.COLOR_BGR2RGB)
plt.imshow(BW, cmap=plt.get_cmap('gray'))
plt.imsave("Q1_1.png", BW, cmap=plt.get_cmap('gray'))
plt.show()


# In[204]:

Q1_2 = np.zeros((width, height, 3), dtype = "uint8")

for x in range(width):
    for y in range(height):
        if BW[x, y] == False:
            Q1_2[x, y, :] = im[x, y, :]
        else:
            Q1_2[x, y, :] = backg[x, y, :]
            
    
plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(Q1_2)
plt.show()


# # Question 2

# In[202]:

'''
math:
1)load image 
2)blur image
3)shapened mask = image - blur
4)final image = image + mask
'''

C = plt.imread("Q2.jpg")

#plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(C)
plt.show()


# In[190]:

img = np.array(C)

blur = cv2.GaussianBlur(img,(5,5),4)

plt.imshow(blur) 
plt.show()


# In[191]:

diff = img - blur

plt.imshow(diff)
plt.show()


# In[192]:

fin = img + diff

plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(fin)
plt.show()


# In[7]:

C = Image.open("Q2.jpg")
out = C.filter(ImageFilter.UnsharpMask(2,200,4))
plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(out)
plt.imsave("Q2_cv2.png", out)
plt.show()


# In[201]:

# save the image files
# fin diff blur
plt.imsave("Q2_blur.png", blur)
plt.imsave("Q2_diff.png", diff)
plt.imsave("Q2_fin.png", fin)


# # Question 3

# # A: Nearest Neighbour

# In[166]:

Q3 = plt.imread("colour.jpg")
im_q3 = np.array(Q3)

plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(im_q3)
plt.show()


# In[167]:

'''
im_q3: input (image)
S:     scale of transformation

out: output (image)
'''
def NN_Resize(inp, S):
    
    # Calculate size of new image
    rows = D.shape[0]
    cols = D.shape[1]
    nRows = round(rows*S) - 1
    nCols = round(cols*S) - 1
    out = np.zeros([nRows, nCols, 3], dtype="uint8")

    for x in range(nRows):
        for y in range(nCols):
            out[x, y] = inp[round(x/S), round(y/S)]
            
    return out


# In[181]:

# run the NN_Resize function
out_q3_1 = NN_Resize(im_q3, 2)
out_q3_2 = NN_Resize(im_q3, 0.05)

plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(out)

plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(out2)
plt.show()


# # save the output files
# plt.imsave('Q3_NN_2.png', out_q3_1)
# plt.imsave('Q3_NN_0.05.png', out_q3_2)

# # B: Bilinear

# In[163]:

def BL_Resize(im_q3, Scale):
    
    
    # Calculate size of new image
    rows = inp.shape[0]
    cols = inp.shape[1]
    nRows = round(rows*Scale) - 1
    nCols = round(cols*Scale) - 1
    
    out = np.zeros([nRows, nCols, 3], dtype="uint8")
    
    for r in range(nRows):
        for c in range(nCols):
            R = r/Scale # approximate position in original image
            C = c/Scale # parsed to int in BL_Interp
            out[r, c] = BL_Interp(inp, R, C, Scale)
            
    return out


# In[162]:

def BL_Interp(im_q3, r, c, Scale):
    rgb = np.zeros((3), dtype="uint8") # reset the rgb array
    
    # find integer/fraction values for interpolation
    Ri = int(r) # cast to int, don't round
    Ci = int(c) # this ensures positive fractions
    Rf = r - Ri
    Cf = c - Ci
    
    Ri = min(Ri, inp.shape[0]-1) # R (int) plus 1
    Ci = min(Ci, inp.shape[1]-1) # C (int) plus 1
    RiP1 = min(Ri+1, inp.shape[0]-1) # R (int) plus 1
    CiP1 = min(Ci+1, inp.shape[1]-1) # C (int) plus 1
    
    for channel in range(im.shape[2]):
        
        bl = inp[Ri  , Ci  , channel]
        br = inp[RiP1, Ci  , channel]
        tl = inp[Ri  , CiP1, channel]
        tr = inp[RiP1, CiP1, channel]
        
        # calculate the interpolation
        top = Rf * tr + (1.-Rf) * tl
        bot = Rf * br + (1.-Rf) * tl
        bil = Cf * top + (1.-Cf) * bot
        
        rgb[channel] = round(bil)
    
    return rgb
    
    


# In[185]:

out_q3_3 = BL_Resize(im_q3, 2)

plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(BL_2)
plt.show()


# plt.imsave("Q3_BI_2.png", out_q3_3)

# # Question 4

# In[3]:

img1 = cv2.imread('NN_2.png',0)          # queryImage
img2 = cv2.imread('BL_2.png',0)          # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

plt.imshow(img3),plt.show()





