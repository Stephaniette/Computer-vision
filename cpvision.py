#Importing the important modules

import streamlit as st
import cv2
import numpy as np
from PIL import Image

#load the image and convert into numpy array
def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)

#convert BGR to RGB (to fix the blue tint)
def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Face Detection
def face_detection(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert the image to gray
    faces = face_cascade.detectMultiScale(grayScale, scaleFactor = 1.3, minNeighbors = 5)

#Draw rectangles around faces
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x + w, y + h), (255, 0, 0),2)
        return convert_bgr_to_rgb(image) #ensure correct color format

#Edge detection function
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

#Gray scale conversion
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Apply the Gaussian blur
def blur(image):
    return cv2.GaussianBlur(image, (15, 15),0)

#Apply the binary thresholding
def threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

    

# Contour Detection
def countour_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Draw contours on the image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    return convert_bgr_to_rgb(image)  #Ensure correct color format


#sharpan the image
def sharpen(image):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return convert_bgr_to_rgb(cv2.filter2D(image, -1, kernel))

# Emboss function
def emboss(image):
    kernel = np.array([[-2,-1,0],[-1,1,1],[0,-1,0]])
    return convert_bgr_to_rgb(cv2.filter2D(image, -1,kernel))


#applying cartoon effect
def cartoonize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    color = cv2.bilateralFilter(image, 9, 300, 300) #smoothen colors
    cartoon_image = cv2.bitwise_and(color, color, mask=edges)
    return convert_bgr_to_rgb(cartoon_image)

#Invert function
def invert(image):
    return convert_bgr_to_rgb(cv2.bitwise_not(image))

#Process image based on the user selection
def process_image(image, operation):
    operations = {
        "Original" : lambda img:img,
        "Face Detection": face_detection,
        "Edge Detection": edge_detection,
        "Grayscale": grayscale,
        "Blur": blur,
        "Threshold": threshold,
        "Contour Detection": countour_detection,
        "Sharpen": sharpen,
        "Embosis": emboss,
        "Cartoonize": cartoonize,
        "Invert": invert
    }
    return operations.get(operation, lambda img:img)(image.copy())

# Main function for streamlit app
def main():
    st.set_page_config(page_title="Image Processing App", page_icon="📷", layout= "wide")

    st.title("Image Processing App")
    st.write("Upload an Image and apply various techniques on image")

    with st.sidebar:
        st.header("Setting")
        uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png','jpeg'])

        #select an operation
        operation = st.selectbox("choose an operation", ['original','Face Detection','Edge Detection','Grayscale','Blur','Threshold','Contour Detection','Sharpem',
        'Embosis','Cartoonize','Invert'])

        #Ensure the image uploaded before processing

        if uploaded_file:
            image = load_image(uploaded_file)

            if st.button("Apply"):
                with st.spinner('Processing ...'):
                    processed_image = process_image(image, operation)
                    st.session_state['Processed_image'] = process_image  #store image in state memory

                #Display processed image
                st.image(processed_image, caption = f'Processed Image ({operation})', use_column_width=True)

            #Enable User to download the image
            if "processed_Image" in st.session_state:
                st.download_button(
                    label = "Download Processed Image",
                    data = cv2.imencode('.png', st.session_state["Processed_image"])[1].tobytes(),
                    file_name ="Processed_Image.png",
                    mime = "image/png"
                )

# Run the App
if __name__ == '__main__':
    main()


###############################################################

 



    
                                                    
