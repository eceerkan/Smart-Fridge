Smart Fridge to keep users informed of the contents of their fridge. 

Upon running "objectdetection.py" on the Raspberry Pi terminal, each triggering of the magnetic switch mounted onto the fridge door activates the camera. Below is an example snapshot with objects detected and labelled:

![F4](https://github.com/eceerkan/Smart-Fridge/assets/105375774/97a04bf2-a8e5-45d4-bf5d-632333eca49a)

The items detected in the snapshot is then used to generate a contents list which is stored in "FridgeContents.json". This information is displayed as follows. Should the estimated expiry date not match what is written on the labelling of the item, the user can manually change the expiry date using the "Edit Row" button.

<img width="441" alt="F7" src="https://github.com/eceerkan/Smart-Fridge/assets/105375774/e11714a7-d515-47d1-9a30-9b7a303504ad">

