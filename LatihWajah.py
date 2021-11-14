import cv2, os, numpy as np
from PIL import Image

wajahDir ='datawajah'
latihDir = 'latihwajah'

def getImageLabel(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []
    for imagePaths in imagePaths:
        PILImg = Image.open(imagePaths).convert('L') #convert kedalam grey atau abu abu
        imgNum = np.array(PILImg,'uint8')
        faceID = int(os.path.split(imagePaths)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y+h,x:x+w])
            faceIDs.append(faceID)
        return faceSamples,faceIDs

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("Mesin Sedang Melakukan Training Data Wajah. Tunggu Dalam Beberapa Detik")
faces,IDs = getImageLabel(wajahDir)
faceRecognizer.train(faces,np.array(IDs))

#SIMPAN
faceRecognizer.write(latihDir+'/training.xml')
print ("Sebanyak {0} data wajah telah ditrainingkan ke mesin.", format(len(np.unique(IDs))))

