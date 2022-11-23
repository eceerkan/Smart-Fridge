from picamera import PiCamera
from time import sleep
import RPi.GPIO as GPIO
import signal
import sys
import cv2

button_pin=27
image =[]
"""
def signal_handler(sig,frame):
        GPIO.cleanup()
        sys.exit(0)
"""
if __name__=='__main__' :
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(button_pin,GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(button_pin,GPIO.FALLING,bouncetime=100)

camera=PiCamera() 
i=0

while True:
        if  GPIO.event_detected(button_pin):
                camera.capture('/home/pi/Project/Cameraimages/images%s.jpg' %(i))
                img=cv2.imread('/home/pi/Project/Cameraimages/images%s.jpg' %(i))
                image.append(img)
                img=cv2.rotate(image[i],cv2.ROTATE_180)
                cv2.imwrite('/home/pi/Desktop/image_rotated%s.jpg' %(i), img)
                print(i)
                i+=1
