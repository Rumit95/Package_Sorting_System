import uvicorn
from fastapi import FastAPI, UploadFile
import numpy as np
import pickle
import cv2
import os
import io
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

orentation_model = tf.keras.models.load_model('model_box_orentation.hdf5')
Box_Orentation_Classes = pickle.load(open("Box_Orentation_Classes.pkl", 'rb'))

side_orentaion_model = tf.keras.models.load_model('model_box_defects_side_view.hdf5')
Box_Side_Defects_Classes = pickle.load(open("Box_Side_Defects_Classes.pkl", 'rb'))

top_orentation_model = tf.keras.models.load_model('model_box_defects_top_view.hdf5')
Box_Top_Defects_Classes = pickle.load(open("Box_Top_Defects_Classes.pkl", 'rb'))

app = FastAPI()


@app.post('/predict')
async def upload_image(file: UploadFile):
    contents = await file.read()

    # Convert the file contents to a PIL Image object
    image = Image.open(io.BytesIO(contents))

    #img = load_img(fname)
    img = img_to_array(image)
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

    if camera==0:
        damage_prediction = side_orentaion_model.predict(resized_image)
        if damage_prediction < 0.5:
            value=0
        else:
            value=1 

    else:
        damage_prediction = top_orentation_model.predict(resized_image)
        if damage_prediction < 0.5:
            value=0
        else:
            value=1
  
    return {'cam':Box_Orentation_Classes.keys()[camera],'condition':Box_Top_Defects_Classes.keys()[value]}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn fapi:app --reload