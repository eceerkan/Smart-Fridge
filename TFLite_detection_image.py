# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
sys.path.append('/usr/lib/python3/dist-packages')
import glob
import importlib.util
from picamera import PiCamera
from time import sleep
import RPi.GPIO as GPIO
import signal
from listcomparison import listcompare
from github import Github

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default=None)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder',
                    action='store_true')
parser.add_argument('--noshow_results', help='Don\'t show result images (only use this if --save_results is enabled)',
                    action='store_false')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()


# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

save_results = args.save_results # Defaults to False
show_results = args.noshow_results # Defaults to True

IM_NAME = args.image
IM_DIR = args.imagedir

# If both an image AND a folder are specified, throw an error
if (IM_NAME and IM_DIR):
    print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
    sys.exit()

# If neither an image or a folder are specified, default to using 'test1.jpg' for image name
if (not IM_NAME and not IM_DIR):
    IM_NAME = 'test1.jpg'

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'


# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.JPG') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
    if save_results:
        RESULTS_DIR = IM_DIR + '_results'

elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)
    if save_results:
        RESULTS_DIR = 'results'

# Create results directory if user wants to save results
if save_results:
    RESULTS_PATH = os.path.join(CWD_PATH,RESULTS_DIR)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

######
#initilisation of camera
camera=PiCamera() 
#initilisation of switch 
button_pin=27
GPIO.setmode(GPIO.BCM)
GPIO.setup(button_pin,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(button_pin,GPIO.FALLING,bouncetime=100)
#i=0 # loop iterator for the switching function

#dictionaries to be used
FridgeOld=dict() #dictionary to hold the old list
FridgeNew=dict() #dictionary to hold the updated list
var=20 #how much variation is allowed from the central point for item tracking

#Recommended use days of each fruit are read from the text file and assigned to a dictionary
ExpirationDays = {}
file = open("ExpirationDays.txt",'r')
for line in file:
    key, value = line.split(':')
    ExpirationDays[key] = (int) (value)

i=0
 #MAIN LOOP   
while True:
# Loop over every image and perform detection
    if  GPIO.event_detected(button_pin):
            print(i+1)
            #retrive github text file data which is the up-to-date website information 
            g = Github("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")
            repo = g.get_repo("eceerkan/Smart-Fridge")
            contents = repo.get_contents("FridgeContents.txt")
            #content = contents.decoded_content

            camera.capture('/home/pi/Project/Smart-Fridge/images/image.jpg')
            image=cv2.imread('/home/pi/Project/Smart-Fridge/images/image.jpg')
            # img=cv2.rotate(image[i],cv2.ROTATE_180)
            #cv2.imwrite('/home/pi/tflite_project/images/image%s.jpg' %(i), image[i])
            #Load image and resize to expected shape [1xHxWx3]            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ =image.shape 
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)
        
            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std
        
            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()
        
            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
        
            detections = []
        
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for m in range(len(scores)):
                if ((scores[m] > min_conf_threshold) and (scores[m] <= 1.0)):
        
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[m][0] * imH)))
                    xmin = int(max(1,(boxes[m][1] * imW)))
                    ymax = int(min(imH,(boxes[m][2] * imH)))
                    xmax = int(min(imW,(boxes[m][3] * imW)))
                    
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        
                    # Draw label
                    object_name = labels[int(classes[m])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[m]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    centerx=((xmax-xmin)/2)
                    centery=((ymax-ymin)/2)
        
                    detections.append([object_name, centerx, centery, scores[m], xmin, ymin, xmax, ymax])
        
            # All the results have been drawn on the image, now display the image
            if show_results:
                cv2.imshow('Object detector', image)
        
            # Save the labeled image to results folder if desired
            if save_results:
                # Get filenames and paths
                image_fn ="image.jpg" 
                image_savepath = os.path.join(CWD_PATH,RESULTS_DIR,image_fn)
                
                base_fn, ext = os.path.splitext(image_fn)
                txt_result_fn = base_fn+'.txt'
                txt_savepath = os.path.join(CWD_PATH,RESULTS_DIR,txt_result_fn)
        
                # Save image
                cv2.imwrite(image_savepath, image)
                with open(txt_savepath,'w') as f:
                                for detection in detections:
                                    f.write('%s %d %d %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5],detection[6],detection[7]))        
                FridgeOld=FridgeNew.copy()
                FridgeNew.clear()
                listcompare(FridgeOld, FridgeNew, detections, var, ExpirationDays)
                
                # Write results to text file
                with open("FridgeContents.txt",'w') as f:
                    for key, value in FridgeNew.items(): 
                        f.write('%s:%s\n' % (key, value))
                        repo.update_file(contents.path, "Updated list", f"{key,value}+\n ", contents.sha)
                f.close()

# Clean up
cv2.destroyAllWindows()
