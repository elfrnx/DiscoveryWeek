import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model(".venv/keras_model.h5", compile=False)

# Load the labels
class_names = open(".venv/labels.txt", "r").readlines()

# Function to make predictions
def make_prediction(image):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize and preprocess the image using Image.resize
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score

# Streamlit App
def main():
    st.title("Image Classifier App")
    st.sidebar.title("Options")

    # Use Webcam
    st.sidebar.markdown("### Webcam")

    # Create a VideoCapture object
    video_capture = cv2.VideoCapture(0)

    # Display the webcam frame in the main area
    frame_placeholder = st.empty()

    run_webcam = st.sidebar.checkbox("Run Webcam")

    # Capture Photo button
    if st.button("Capture Photo"):
        _, captured_frame = video_capture.read()

        # Convert BGR to RGB
        captured_frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image format
        captured_image = Image.fromarray(captured_frame_rgb)

        # Resize and preprocess the image
        captured_image = captured_image.resize((224, 224))
        captured_image_array = np.asarray(captured_image)
        normalized_captured_image_array = (captured_image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_captured_image_array

        # Make a prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Draw bounding box and text on the captured frame
        cv2.putText(captured_frame_rgb, f"{class_name}: {confidence_score:.2%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the captured frame
        st.image(captured_frame_rgb, channels="RGB", caption="Captured Photo", use_column_width=True)

    while run_webcam:
        _, frame = video_capture.read()

        # Check if the frame is empty (None)
        if frame is None:
            st.error("Failed to capture frame from the webcam.")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image format
        image_pil = Image.fromarray(frame_rgb)

        # Resize and preprocess the image
        image_pil = image_pil.resize((224, 224))
        image_array = np.asarray(image_pil)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make a prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Draw bounding box and text on the frame
        cv2.putText(frame_rgb, f"{class_name}: {confidence_score:.2%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the webcam frame
        frame_placeholder.image(frame_rgb, channels="RGB", caption="Webcam Feed", use_column_width=True)

    # Release the VideoCapture object when done
    video_capture.release()

if __name__ == "__main__":
    main()

