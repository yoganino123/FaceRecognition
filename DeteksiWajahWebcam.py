'''
install modul opencv
pip install opencv-contrib-python
python -m pip install --upgrade opencv-contrib-python
pip install Pillow
deteksi wajah https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascades_frontalface_default.xml
deteksi mata https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascades_eye.xml
langkah untuk face recognition: rekam data wajah, recognition
'''
import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 640) #ubah lebar cam
cam.set(4, 480) #ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')

while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5 ) #frame, scalefactor, minneighbour
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2)
        roiAbuAbu = abuAbu[y:y+h,x:x+w]
        roiWarna = frame[y:y+h,x:x+w]
        eyes = eyeDetector.detectMultiScale(roiAbuAbu)
        for (xe,ye,we,he) in eyes:
            cv2.rectangle(roiWarna, (xe,ye),(xe+we,ye+he),(0,0,255),1)

    cv2.imshow('WebcamKu',frame)
    #cv2.imshow('WebcamKu 2', abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()