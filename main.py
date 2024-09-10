from textblob import TextBlob
import pandas as pd
import altair as at
import numpy as np
import streamlit as st
import cleantext
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sklearn
from PIL import Image

# Load the emotion prediction model
pipe_lr = joblib.load(open("C:/Users/hari1/Downloads/Sentimental Analysis/model/text_emotion.pkl", "rb"))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Dictionary for emoji representation
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
    "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Function to predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Function to get the prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def get_emotion():
    emotions = {
        "happy": np.random.uniform(0.5, 1.0),  # Random confidence score for demo
        "sad": np.random.uniform(0.0, 0.5),
        "neutral": np.random.uniform(0.0, 0.5)
    }
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    top_emotion = sorted_emotions[0]  # Get the highest confidence emotion
    return top_emotion[0], top_emotion[1]

# Function to detect faces and annotate them with emotions and percentages
def detect_faces_and_emotions(image):
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    emotions = []
    for (x, y, w, h) in faces:
        # Get simulated emotion and confidence score
        detected_emotion, confidence = get_emotion()
        emotions.append({"box": (x, y, w, h), "emotion": detected_emotion, "confidence": round(confidence*100,2)})
        
        # Draw rectangle and emotion label with percentage
        color = (0, 255, 0) if detected_emotion == "happy" else (255, 0, 0)
        cv2.rectangle(img_array, (x, y), (x + w, y + h), color, 2)
        label = f"{detected_emotion}: {confidence * 100:.1f}%"
        cv2.putText(img_array, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img_array, emotions

st.header('Sentiment Analysis')

# Text analysis section
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        col1, col2 = st.columns(2)
        prediction = predict_emotions(text)
        probability = get_prediction_proba(text)
        with col1:
            st.success("Prediction:")
            st.write('Polarity: ', round(blob.sentiment.polarity, 2))
            st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))
            if round(blob.sentiment.polarity, 2) > 0:
                st.write("Sentiment:\n Positive")
            elif round(blob.sentiment.polarity, 2) < 0:
                st.write("Sentiment:\n Negative")
            else:
                st.write("Sentiment:\n Neutral")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"{prediction}: {emoji_icon}")
            st.success(f"Confidence: {round(np.max(probability),4)}")
        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]
            fig = at.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)
        st.write('Clean Text')
        st.write(cleantext.clean(text, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))


# CSV file analysis section
with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    # Function to calculate sentiment score
    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    # Function to assign sentiment category based on score
    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    # If a CSV is uploaded, process and analyze
    if upl:
        df = pd.read_excel(upl)
        del df['Unnamed: 0']  # Remove the unnamed column (if present)
        df['score'] = df['tweets'].apply(score)  # Apply the score function to calculate polarity
        df['analysis'] = df['score'].apply(analyze)  # Apply the analysis function to categorize sentiment

        # Show the first few rows of the dataframe
        st.write(df.head(10))

        # Visualization: Count of each sentiment
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='analysis', data=df, ax=ax, palette="Set2")
        st.pyplot(fig)

        # Visualization: Pie chart for sentiment distribution
        st.subheader("Sentiment Pie Chart")
        sentiment_counts = df['analysis'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
        st.pyplot(fig2)

        # Prepare DataFrame for download (after showing visualizations)
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        # Button to download the CSV file after visualizations
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment_analysis_results.csv',
            mime='text/csv',
        )
with st.expander('Analyze Image'):
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file)
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        # Perform face and emotion detection
        img_with_faces, detected_emotions = detect_faces_and_emotions(image)

        # Display results
        st.success(f"Detected Emotions: {detected_emotions}")
        st.image(img_with_faces, caption='Image with Detected Faces and Emotions (with percentages)', use_column_width=True)

        # Optional: Add a download button to save the processed image with highlighted emotions
        processed_image_pil = Image.fromarray(cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB))
        st.download_button(
            label="Download Image with Emotions",
            data=processed_image_pil.tobytes(),
            file_name="emotion_image_with_percentage.png",
            mime="image/png"
        )


# Render additional page content (if needed)
#imagepage.renderPage()
