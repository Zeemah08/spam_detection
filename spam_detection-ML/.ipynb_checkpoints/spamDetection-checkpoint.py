import pandas as pd
import streamlit as st
import joblib
import json

MAX_SIZE_BYTES = 500 * 1024  # 500 KB

model = joblib.load("best_pipeline.pkl")
with open("model_metrics.json", "r") as f:
    metrics = json.load(f)


st.markdown("<h2 style = 'text-align: left; color: orange;'> Email Spam Detector </h2>", unsafe_allow_html = True)
st.markdown("<h3 style = 'text-align: left; color: gray;'> Enter a message below and click 'Classify' to check if it is a spam or not </h3>", unsafe_allow_html = True)
st.divider()


message = ""
uploaded_file = st.file_uploader("Upload a text file (maximum 500 KB)", type = ["txt"])

# Model evaluation
metrics_df = pd.DataFrame(metrics).transpose()
metrics_df_filtered = metrics_df[["precision", "recall", "f1-score", "support"]]
metrics_df_filtered = metrics_df_filtered.rename(index= {"1" : "Spam", "0" : "Not Spam"})
metrics_df_filtered = metrics_df_filtered.loc[["Spam", "Not Spam"]]


with st.sidebar:
    st.subheader("Disclaimer")
    st.markdown("<h2 style = 'color: red;'>This is a machine learning demonstration for educational purposes.\n It is not a professional spam filter. Messages entered are not stored </h2>", unsafe_allow_html = True)

    st.header("Model Details")
    st.subheader("Loaded Model Pipeline")
    st.write(model.steps)

    st.subheader("Model Settings")
    custom_threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown(f"**Custom Threshold:** {custom_threshold}")

    st.subheader("Model Performance")
    st.markdown(f"**Overall Accuracy:** {metrics['accuracy']:.2f}")
    st.dataframe(metrics_df_filtered.round(2))


if uploaded_file is not None:
    if uploaded_file.size > MAX_SIZE_BYTES:
        st.error("The file is too large. Please upload a file of 500 KB or less.")
    else:
        message = uploaded_file.getvalue().decode("utf-8", errors = "replace")
        st.info("File uploaded successfully")

user_input = st.text_area("Or Enter Message Here", height = 150, value = message, max_chars = MAX_SIZE_BYTES)


if st.button("Classify"):
    if user_input.strip():
        st.info("Analyzing message...")
        prediction = model.predict_proba([user_input])

        spam_confidence = prediction[0][1]
        ham_confidence = prediction[0][0]

        label = "🚫 Spam" if spam_confidence >= custom_threshold else "✅ Not Spam"

        if label == "🚫 Spam":
            st.error(label)
        else:
            st.success(label)

        with st.expander("Show Detailed Probabilities"):
            st.write(f"Confidence for Spam: {round(spam_confidence * 100)}%")
            st.write(f"Confidence for Not Spam: {round(ham_confidence * 100)}%")
            st.markdown("Classification is based on a custom threshold")
    else:
        st.warning("Please enter a message to classify")