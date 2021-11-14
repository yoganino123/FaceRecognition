'''
install modul opencv
pip install opencv-contrib-python
python -m pip install --upgrade opencv-contrib-python
pip install Pillow
'''
import cv2
cam = cv2.VideoCapture(0)
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('WebcamKu',frame)
    cv2.imshow('WebcamKu 2', abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()