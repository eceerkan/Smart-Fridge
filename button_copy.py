
from time import sleep
import RPi.GPIO as GPIO
from github import Github



button_pin=27
if __name__=='__main__' :
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(button_pin,GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(button_pin,GPIO.FALLING,bouncetime=100)

# Connect to GitHub

while True:
  if  GPIO.event_detected(button_pin):
    g = Github("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")
    repo = g.get_repo("eceerkan/Smart-Fridge")
    contents = repo.get_contents("transfer.txt")
    count = int(contents.decoded_content)
    count += 1
    repo.update_file(contents.path, "Updated count", str(count), contents.sha)
    print(contents)
