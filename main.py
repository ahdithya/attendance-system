# import module
import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
import pickle

from PIL import Image
from datetime import datetime
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

# initialize detector and embedder
embedder = FaceNet()
detector = MTCNN()

model = pickle.load(open("./packages/model.pkl", 'rb'))
encoder = pickle.load(open("./packages/encoder.pkl", 'rb'))
threshold = 0.65


# get embeddings 
def get_embeddings(face_img):
    face_img = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(face_img)
    return embedding[0]

# loag image
# def load_image(image_path):
#     image = cv.imread(image_path)
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     return image

# detect face and extract faces
def extract_faces(img):
    # img = load_image(img_path)
    faces = detector.detect_faces(img)
    embeddings = []
    boxes = []
    if faces:
        for face in faces:
            box = face['box']
            x, y, w, h = box
            face_img = img[y:y+h, x:x+w]
            face_img = cv.resize(face_img, (160, 160))
            embedding = get_embeddings(face_img)
            embeddings.append(embedding)
            boxes.append(box)
    return embeddings, boxes

# prediction 
def predict(image):
    embeddings, boxes = extract_faces(image)
    predictions = []
    if embeddings:
        for embedding in embeddings:
            proba = model.predict_proba(embedding.reshape(1, -1))
            max_proba = np.argmax(proba)
            if proba[0][max_proba] > threshold:
                predictions.append(encoder.inverse_transform([max_proba])[0])
            else:
                print("Unknown")
                predictions.append("Unknown")   
    return predictions, boxes

# Attendance
def attendance(df, path):
    predictions, boxes = predict(path)
    for prediction in predictions:
        if prediction in df["name"].values:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.loc[df['Name'] == prediction, 'Attendance'] = 'present'
            df.loc[df['Name'] == prediction, 'Time'] = now
        else:
            print("Data Tidak Ditemukan")
    return df, predictions, boxes

# Initialize dataframe to store attendance
df = pd.DataFrame({
        'Name': ["Aditya", "Elon Musk", "Fitriah"],
        'Attendance': [""] * 3,
        'Time': [""] * 3
    })


# initialize streamlit
st.set_page_config(page_title="Attendance System")
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 20px '>Attendance System</h1>",
    unsafe_allow_html=True,)
# Take a picture using the webcam
if "image" not in st.session_state:
        st.session_state["image"] = None
# tabs 
tab1, tab2 = st.tabs(["Upload Image", "Webcam"])

with tab1:
    st.markdown("<h3 style='text-align: center; padding-bottom: 0px;'>Upload Image</h3>", unsafe_allow_html=True)
    # Add the uploaded image to session state
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="upload_image")
    if uploaded_file is not None:
        img = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)
        st.session_state["image"] = img
        
        df, predictions, boxes = attendance(df, img)
        for (box, prediction) in zip(boxes, predictions):
            x, y, w, h = box
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv.putText(img, prediction, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert image back to PIL for Streamlit display
        image_with_boxes = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

       
       
# with tab2: 
#     st.markdown("<h3 style='text-align: center; padding-bottom: 0px;'>Take A Picture</h3>", unsafe_allow_html=True) 
            
#     picture = st.camera_input("Take a picture")

#     # Add the captured image to session state
#     if picture is not None:
#         bytes_data = picture.getvalue()
#         img = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
#         st.session_state["image"] = img
    

# if st.button("Mark Attendance"):
#     df = attendance(df, st.session_state["image"])    
    


st.markdown("<h2 style='text-align: center; padding-bottom: 0px;'>Result</h2>", unsafe_allow_html=True)
    
with st.sidebar:
    st.markdown("<h2 style='text-align:center; padding-bottom: 20px;'>Attendance</h2>", unsafe_allow_html=True)
    st.dataframe(df)