import cv2
import sys
import os.path
import json
import base64 
import webbrowser
import requests
import urllib3

def askgoogle(filename):
 filePath = filename
 searchUrl = 'http://www.google.com/searchbyimage/upload'
 multipart = {'encoded_image': (filePath, open(filePath, 'rb')), 'image_content': ''}
 response = requests.post(searchUrl, files=multipart, allow_redirects=False)
 fetchUrl = response.headers['Location']
 webbrowser.open(fetchUrl)

def detect(filename, cascade_file = "../lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    
    i = 0
    crop_img = image
    for (x, y, w, h) in faces:
    	i+=1;
    	if i == 1:
    	 crop_img = image[y:y+h, x:x+w]
    	 
    	 
    #cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(1)
    cv2.imwrite("out.png", crop_img)
    askgoogle("out.png")

if len(sys.argv) != 2:
    sys.stderr.write("usage: detect.py <filename>\n")
    sys.exit(-1)
    
detect(sys.argv[1])
