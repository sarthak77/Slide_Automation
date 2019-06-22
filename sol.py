import sys
import os
import cv2
import numpy as np


if __name__=='__main__':

    #error handling if proper arguments not given
    if len(sys.argv)!=3:
        print("Usage: <rollno>.py <path/to/slides/directory> <path/to/frames/directory>")
    
    #assigning directory values
    d1=sys.argv[1]
    d2=sys.argv[2]

    #appending images of slides
    slides=[]
    for filename in os.listdir(d1):
        slides.append(os.path.abspath(d1+"/"+filename))

    #appending images of frames
    images=[]
    for filename in os.listdir(d2):
        images.append(os.path.abspath(d2+"/"+filename))

    #stores ppt which the image represents 
    imgarr=[]
    
    #temporary array for storing max good points
    gparr=[]

    #initialising arrays
    for i in range(len(images)):
        gparr.append(-1)
        imgarr.append("")

    #running loop for images
    imgcount=-1#for appending in the array
    sift = cv2.xfeatures2d.SIFT_create()#part of sift

    for slide in slides:
        imgcount=-1#starting from initial value again
        
        I1=cv2.imread(slide,0)#read
        kp_1,desc_1=sift.detectAndCompute(I1, None)#find kp

        for img in images:
            
            imgcount=imgcount+1#increment to next image

            I2=cv2.imread(img,0)#read
            kp_2,desc_2=sift.detectAndCompute(I2, None)#sift

            #part of sift algorithm
            index_params=dict(algorithm=0, trees=5)
            search_params=dict()
            flann=cv2.FlannBasedMatcher(index_params, search_params)
            matches=flann.knnMatch(desc_1, desc_2, k=2)

            #calculate good points
            good_points=0
            ratio=0.6
            for m,n in matches:
                if m.distance < ratio*n.distance:
                    good_points=good_points+1

            a=good_points
            #check if more no of good points matched
            if(a>gparr[imgcount]):
                gparr[imgcount]=a
                imgarr[imgcount]=slide

    #writing to a file
    outputfile=open("20171091_20171059.txt","w")
    for i in range(len(imgarr)):
        print(images[i], end=" ", file=outputfile)
        print(imgarr[i], end="\n", file=outputfile)
    outputfile.close()
