from picamera import PiCamera
from time import sleep
import RPi.GPIO as GPIO
import signal
import sys

def signal_handler(sig,frame):
        GPIO.cleanup()
        sys.exit(0)


#def my_callback(channel):
      #  print("something")
#       return 1

if __name__=='__main__' :
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(27,GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(27,GPIO.FALLING,bouncetime=100)

camera=PiCamera()
i=0

while True:
        if  GPIO.event_detected(27):
                i=i+1
                camera.capture('/home/pi/Desktop/image%s.jpg' %(i))
                print("something")


