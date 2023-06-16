import pandas as pd
import numpy as np
import cv2
import pickle
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

orentation_model = tf.keras.models.load_model('model_box_orentation.hdf5')
Box_Orentation_Classes = pickle.load(open("Box_Orentation_Classes.pkl", 'rb'))

side_orentaion_model = tf.keras.models.load_model('model_box_defects_side_view.hdf5')
Box_Side_Defects_Classes = pickle.load(open("Box_Side_Defects_Classes.pkl", 'rb'))

top_orentation_model = tf.keras.models.load_model('model_box_defects_top_view.hdf5')
Box_Top_Defects_Classes = pickle.load(open("Box_Top_Defects_Classes.pkl", 'rb'))



#fname = r"C:\Users\Rumit\Downloads\Wonderbiz\BlenderDeepLearning\POC2\POC2\Box_condition\Box_Top_Condition\intact\0407105940330_top.png"
#fname = r"C:\Users\Rumit\Downloads\Wonderbiz\BlenderDeepLearning\POC2\POC2\Box_condition\Box_Top_Condition\damaged\0429090229299_top.png"
#fname = r"C:\Users\Rumit\Downloads\Wonderbiz\BlenderDeepLearning\POC2\POC2\Box_condition\Box_Side_Condition\intact\0681342689666_side.png"
fname = r"C:\Users\Rumit\Downloads\Wonderbiz\BlenderDeepLearning\POC2\POC2\Box_condition\Box_Side_Condition\damaged\0274769934484_side.png"

img = load_img(fname)
img = img_to_array(img)
print("Original image shape:", img.shape)

# Resize the image
resized_image = cv2.resize(img, (380,224))
print("Resized image shape:", resized_image.shape)

# Normalize the image
resized_image = resized_image / 255.0

# Add batch and channel dimensions
resized_image = np.expand_dims(resized_image, axis=-1)
resized_image = np.expand_dims(resized_image, axis=0)

# Make a prediction with the model
prediction = orentation_model.predict(resized_image)
#predicted_class = np.argmax(prediction)
#print(prediction)

if prediction < 0.5:
    camera=0
else:
    camera=1
print('Box is located in ',Box_Orentation_Classes.keys()[camera],' camera angle')

if camera==0:
    damage_prediction = side_orentaion_model.predict(resized_image)
    if damage_prediction < 0.5:
        value=0
    else:
        value=1
    #print(Box_Side_Defects_Classes.keys())
    print('Box Condition is ',Box_Side_Defects_Classes.keys()[value])   

else:
    damage_prediction = top_orentation_model.predict(resized_image)
    if damage_prediction < 0.5:
        value=0
    else:
        value=1
    #print(Box_Top_Defects_Classes.keys())
    print('Box Condition is ',Box_Top_Defects_Classes.keys()[value])