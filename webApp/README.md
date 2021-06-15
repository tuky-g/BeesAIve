# BeesAIve web app

The aim of BeesAIve is to create an alert system that warns the beekeepers if any outsider species are threatening the hive. Our opensource tool helps to prevent the death of the bees by means of an early warning system.

This web app is just a visual proof of concept of the model and use 

Inspired by Saturdays.ai


###How to run it locally
* Copy the deep learning model (extension .h5) in the model folder
* Change the model's name in the main.py file, where the model's path is declared </br>
```model = load_model('models/bs_xception_model.h5')```
  
* Run and create your environment with the requirements file</br>
``$ python3 -m pip install -r requirements.txt``
  
* Start the app: ``$ python3 main.py``



