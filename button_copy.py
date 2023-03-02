from time import sleep
import RPi.GPIO as GPIO
from github import Github



button_pin=27
if __name__=='__main__' :
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(button_pin,GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(button_pin,GPIO.FALLING,bouncetime=100)

# Connect to GitHub
g = Github("your-username", "your-password")
repo = g.get_repo("your-username/your-repo-name")

# Get the contents of the text file in the repository
contents = repo.get_contents("transfer.txt")

count = int(contents.decoded_content)

while True:
        if  GPIO.event_detected(button_pin):
          count += 1
          # Update the text file in the repository
          repo.update_file(contents.path, "Updated count", str(count), contents.sha)
                
