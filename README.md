
# Box Condition Prediction

This project implements a Box Condition Prediction system using deep learning algorithms. It includes two trained models: one for detecting the camera angle (orientation) of the box and another for determining whether the box is damaged or not.

The project utilizes the FastAPI and Streamlit frameworks for creating a web application that allows users to upload an image of a box and receive predictions about its camera angle and condition.

The deployed web application can be accessed at: [Box Condition Prediction Web App](https://rumit95-package-sorting-system-webapp-9m9t5i.streamlit.app/)

## Deep Learning Algorithms

### 1. Camera Angle Detection

The first deep learning algorithm is trained to detect the camera angle or orientation of the box. It uses the following model and classes:

- Model: `model_box_orentation.hdf5`
- Classes: `Box_Orentation_Classes.pkl`

The model takes as input a resized and normalized image of the box and predicts the camera angle, classifying it as either 0 (side view) or 1 (top view).

### 2. Box Condition Detection

The second deep learning algorithm determines whether the box is damaged or not. It uses the following model and classes:

- Model: `model_box_defects_side_view.hdf5` (for side view) and `model_box_defects_top_view.hdf5` (for top view)
- Classes: `Box_Side_Defects_Classes.pkl` (for side view) and `Box_Top_Defects_Classes.pkl` (for top view)

Depending on the camera angle predicted by the first algorithm, the corresponding model is used for box condition detection. The model takes a resized and normalized image of the box and predicts the condition, classifying it as either 0 (undamaged) or 1 (damaged).

## Web Application

The Box Condition Prediction system is implemented as a web application using the following technologies:

- FastAPI: The FastAPI framework is used to create an API endpoint `/predict` that receives an uploaded image file and returns the predictions.
- Streamlit: The Streamlit framework is used to create a user-friendly interface for the web application. It allows users to upload an image file and displays the original image along with the predicted camera angle and box condition.
- Streamlit Deployment: The project is deployed using Streamlit sharing, and the deployed web application can be accessed at [Box Condition Prediction Web App](https://rumit95-package-sorting-system-webapp-9m9t5i.streamlit.app/)

## Requirements

The project has the following dependencies:

- streamlit==0.88.0
- requests==2.26.0
- json==2.0.9
- time==1.8.1
- opencv-python-headless==4.5.3.56
- pillow==8.3.2
- tensorflow==2.6.0
- uvicorn==0.15.0
- fastapi==0.74.0

You can install these dependencies by running the following command:

        pip install -r requirements.txt

## Usage

To run the Box Condition Prediction web application locally, follow these steps:

1. Install the project dependencies using the command mentioned above.
2. Ensure that you have the trained models (`model_box_orentation.hdf5`, `model_box_defects_side_view.hdf5`, `model_box_defects_top_view.hdf5`) and classes files (`Box_Orentation_Classes.pkl`, `Box_Side_Defects_Classes.pkl`, `Box_Top_Defects_Classes.pkl`) in the same directory as the project.
3. Run the following command to start the FastAPI server:

        uvicorn fapi:app --reload

This will start the server at `http://127.0.0.1:8000`.

4. In a separate terminal, run the following command to start the Streamlit web application:

        streamlit run webapp.py


5. Access the web application in your browser at `http://localhost:8501` or as mentioned in the Streamlit sharing link.

- Upload an image file (JPG, JPEG, or PNG) using the file uploader in the sidebar.
- The application will display the original image, along with the predicted camera angle and box condition.

Please note that the deployed web application is accessible at [Box Condition Prediction Web App](https://rumit95-package-sorting-system-webapp-9m9t5i.streamlit.app/) and can be used without running it locally.

## Contact

For any issues or questions regarding the Box Condition Prediction project or web application, please contact the project developer at <rumit.pthr@gmail.com>.



