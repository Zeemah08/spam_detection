import pandas as pd
import streamlit as st
import joblib
import json

MAX_SIZE_BYTES = 500 * 1024  # 500 KB

# ---------- Button colours (change the hex codes here if you like) ----------
CLASSIFY_COLOR = "#1F77B4"        # blue
CLASSIFY_HOVER = "#17608F"
CLEAR_COLOR = "#5A6272"           # grey
CLEAR_HOVER = "#444B58"

# ---------- Load the model and its saved performance figures ----------
model = joblib.load("best_pipeline.pkl")
with open("model_metrics.json", "r") as f:
    metrics = json.load(f)

# ---------- Session defaults ----------
if "message_box" not in st.session_state:
    st.session_state.message_box = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "upload_status" not in st.session_state:
    st.session_state.upload_status = ""
if "upload_ok" not in st.session_state:
    st.session_state.upload_ok = True


def load_uploaded_file():
    """Runs when a file is uploaded: checks its size and puts its text in the message box."""
    uploaded = st.session_state.get(
        f"uploader_{st.session_state.uploader_key}")
    if uploaded is None:
        return
    if uploaded.size > MAX_SIZE_BYTES:
        st.session_state.upload_ok = False
        st.session_state.upload_status = "The file is too large. Please upload a file of 500 KB or less."
        return
    st.session_state.upload_ok = True
    st.session_state.upload_status = "File uploaded successfully"
    st.session_state.message_box = uploaded.getvalue().decode("utf-8", errors="replace")


def clear_all():
    """Runs when Clear is pressed: empties the message box and the file uploader."""
    st.session_state.message_box = ""
    st.session_state.upload_status = ""
    st.session_state.upload_ok = True
    st.session_state.uploader_key += 1  # a new key gives a fresh, empty uploader


# ---------- Button styling ----------
st.markdown(
    f"""
    <style>
    .st-key-classify_btn {{ display: flex; justify-content: flex-end; }}
    .st-key-classify_btn button {{ background-color: {CLASSIFY_COLOR}; border: none; }}
    .st-key-classify_btn button:hover {{ background-color: {CLASSIFY_HOVER}; }}
    .st-key-classify_btn button p {{ color: white; }}
    .st-key-clear_btn button {{ background-color: {CLEAR_COLOR}; border: none; }}
    .st-key-clear_btn button:hover {{ background-color: {CLEAR_HOVER}; }}
    .st-key-clear_btn button p {{ color: white; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2 style = 'text-align: left; color: orange;'> Email Spam Detector </h2>",
            unsafe_allow_html=True)
st.markdown("<h3 style = 'text-align: left; color: gray;'> Enter a message below and click 'Classify' to check if it is a spam or not </h3>", unsafe_allow_html=True)
st.divider()

# ---------- Model evaluation table ----------
metrics_df = pd.DataFrame(metrics).transpose()
metrics_df_filtered = metrics_df[[
    "precision", "recall", "f1-score", "support"]]
metrics_df_filtered = metrics_df_filtered.rename(
    index={"1": "Spam", "0": "Not Spam"})
metrics_df_filtered = metrics_df_filtered.loc[["Spam", "Not Spam"]]

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Disclaimer")
    st.markdown("<h2 style = 'color: red;'>This is a machine learning demonstration for educational purposes.\n It is not a professional spam filter. Messages entered are not stored </h2>", unsafe_allow_html=True)

    st.header("Model Details")
    st.subheader("Loaded Model Pipeline")
    st.write(model.steps)

    st.subheader("Model Settings")
    custom_threshold = st.slider(
        "Classification Threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown(f"**Custom Threshold:** {custom_threshold}")

    st.subheader("Model Performance")
    st.markdown(f"**Overall Accuracy:** {metrics['accuracy']:.2f}")
    st.dataframe(metrics_df_filtered.round(2))

# ---------- Input ----------
st.file_uploader(
    "Upload a text file (maximum 500 KB)",
    type=["txt"],
    key=f"uploader_{st.session_state.uploader_key}",
    on_change=load_uploaded_file,
)
if st.session_state.upload_status:
    if st.session_state.upload_ok:
        st.info(st.session_state.upload_status)
    else:
        st.error(st.session_state.upload_status)

user_input = st.text_area("Or Enter Message Here",
                          height=150, key="message_box", max_chars=MAX_SIZE_BYTES)

# ---------- Buttons: Clear on the left, Classify on the right ----------
col_clear, col_classify = st.columns([6, 1], gap="small")
with col_clear:
    st.button("Clear", key="clear_btn", on_click=clear_all)
with col_classify:
    classify_clicked = st.button("Classify", key="classify_btn")

# ---------- Classification ----------
if classify_clicked:
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
            st.write(
                f"Confidence for Not Spam: {round(ham_confidence * 100)}%")
            st.markdown("Classification is based on a custom threshold")
    else:
        st.warning("Please enter a message to classify")
