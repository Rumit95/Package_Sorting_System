import streamlit as st
import requests
import json
import time
import os

def main():
    st.set_page_config(layout="wide", page_title="Box Condition Prediction")
    st.markdown("<h1 style='text-align: center; color: white;'>Box condition Prediction</h1>", unsafe_allow_html=True)
    st.sidebar.write("## Upload  :gear:")
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)
    col2.markdown("<h3 style='text-align: center; color: white;'>Predicted Output</h3>", unsafe_allow_html=True)

    
    # Create a file uploader in Streamlit

    if uploaded_file is None:
        # Using a path relative to the current file
        current_file_path = os.path.abspath(__file__)
        image_path = os.path.join(os.path.dirname(current_file_path), 'Temp', 'noimg.png')
        col1.image(image_path, caption="Uploaded Image", use_column_width=True)

    else:

        # Display image
        #col1.write("Original Image :camera:")
        col1.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Prepare the payload for FastAPI server
        files = {"file": uploaded_file}
        # Make a POST request to the FastAPI server
        start_time = time.time()
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
            st.sidebar.success("File uploaded successfully!")
            response_data = json.loads(response.text)
            col2.write('\n')
            col2.markdown(f"<h6 style='text-align:center ; color: white;'>Camera angle is {response_data['cam']} view. </h6>", unsafe_allow_html=True)
            col2.markdown(f"<h6 style='text-align:center ; color: white;'>Box Condition is {response_data['condition']}. </h6>", unsafe_allow_html=True)
            execution_time = time.time() - start_time
            st.sidebar.write(f"Time Required {execution_time:.2f} secs.")
        else:
            st.error("Upload failed.")

if __name__ == "__main__":
    main()