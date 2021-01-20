from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg



namelist  = ['centerlight','glasses','happy','leftlight','noglasses','normal' , 'rightlight',
             'sad','sleepy','surprised']
j = 1
imageno = 0
pixval1 = []

matrixA = np.zeros((150,45045))
p = 'subject01.centerlight.pgm'
im = Image.open(p)
#im.show()
for j in range(1,10):
    for i in namelist:
        p = "subject0"+str(j)+"." + i + ".pgm"
        im = Image.open(p)
        pix_val = list(im.getdata())
        matrixA[imageno] = np.array([pix_val])
        imageno = imageno + 1

for j in range(10,16):
    for i in namelist:
        p = "subject"+str(j)+"." + i + ".pgm"
        im = Image.open(p)
        pix_val = list(im.getdata())
        matrixA[imageno] = np.array([pix_val])
        imageno = imageno + 1

matrixB = np.zeros((150,150))
matrixB = np.dot(matrixA ,matrixA.transpose())


eigenValues,eigenVectors = linalg.eig(matrixB)
idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]


requiredVectors = np.zeros((45045,45045))
for i in range(150):
    requiredVectors[i] = np.dot(matrixA.transpose() ,eigenVectors[i])
    
#im = Image.open('subject01.centerlight.pgm')
#im2 = Image.open('subject01.glasses.pgm')
#-----------------------------------------important--------------------
for i in range(15):
    
    k = requiredVectors[i].reshape(231,195)
        
    #pix_val = list(im.getdata()) 
    #pix_val = np.array(pix_val)
    #k = pix_val.reshape(231,195)
    #plt.imshow(k,cmap = plt.cm.gray)
    #plt.show()

#--Expressing as a linear combination.
eigenFaces = np.zeros((15,45045))

for i in range(15):
    eigenFaces[i] = requiredVectors[i]
    
p = 'subject02.wink.pgm'
im = Image.open(p)
pix_val = list(im.getdata())
pix_val = np.array(pix_val)

pr = np.linalg.pinv(eigenFaces.transpose())
ans1 = np.dot(pr,pix_val)

ans2 = np.dot(eigenFaces.transpose(),ans1)
k = ans2.reshape(231,195)

plt.imshow(k,cmap = plt.cm.gray)
plt.show()
