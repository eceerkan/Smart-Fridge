# Smart-Fridge
Smart fridge RPI project for final year design project

What is currently being used for the project:
TFLite_detection_image: Main file, has the main loop that detects objects, makes list comparisons and communicates with the user (email communication has not been completed yet)
listcomparison.py: Has the function that compares the old contents with the newly detected contents and generates a new contents list 
ExpirationDays.txt: Hs the recommended days fruits/vegtables should be used in days
images folder: captured image upon the switch being triggered is saved here and read from here
images_results: 
custom_model_lite folder: Fine-tuned model, is called from the main function