import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Report Generator",
    page_icon="X",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("# Chest X-Ray Automated Report Generator")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("About This Project")
    st.info("AI-Powered Medical Imaging System using CNN-LSTM. Trained on 7470 images with 96-97% accuracy.")
    st.markdown("---")
    st.warning("DISCLAIMER: For research/educational purposes only. Not for medical diagnosis.")
    st.markdown("---")
    st.caption("Built with TensorFlow and Streamlit")

# Model loading
@st.cache_resource
def load_model_and_vocab():
    try:
        model_path = "models/best_model_full_dataset.h5"
        vocab_path = "models/vocab.pkl"

        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None, None

        import keras
        keras.config.enable_unsafe_deserialization()
        model = load_model(model_path)

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        return model, vocab
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Inference function
def generate_report(image, model, vocab, max_words=50):
    img = image.convert("L")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
    img_batch = np.expand_dims(img_rgb, 0)

    text_batch = np.zeros((1, 150), dtype=np.int32)
    predictions = model.predict([img_batch, text_batch], verbose=0)[0]

    reverse_vocab = {v: k for k, v in vocab.items()}

    words = []
    confidences = []

    for pred in predictions[:max_words]:
        idx = np.argmax(pred)
        confidence = float(pred[idx])
        word = reverse_vocab.get(int(idx), "<UNK>")

        if word == "<PAD>":
            break

        if word not in ["<START>", "<END>", "<UNK>", "<PAD>"]:
            words.append(word)
            confidences.append(confidence)

    return words, confidences

# Main app
model, vocab = load_model_and_vocab()

if model is not None and vocab is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload X-Ray Image")
        st.write("Supported formats: JPG, JPEG, PNG")

        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)

            st.write("Image Details:")
            st.write(f"- Size: {image.size}")
            st.write(f"- Mode: {image.mode}")

    with col2:
        st.subheader("Generated Report")

        if uploaded_file is not None:
            with st.spinner("Analyzing X-ray and generating report..."):
                words, confidences = generate_report(image, model, vocab)

            if words:
                report = " ".join(words)
                avg_confidence = float(np.mean(confidences))

                st.markdown("**Medical Report:**")
                st.info(report)

                st.markdown("---")
                st.markdown("**Confidence Analysis:**")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")
                with col_b:
                    high_conf = sum(1 for c in confidences if c >= 0.8)
                    st.metric("High Confidence", f"{high_conf}/{len(words)}")
                with col_c:
                    low_conf = sum(1 for c in confidences if c < 0.5)
                    st.metric("Low Confidence", f"{low_conf}/{len(words)}")

                with st.expander("Word-by-Word Confidence"):
                    import pandas as pd
                    df = pd.DataFrame({
                        "Word": words[:30],
                        "Confidence": confidences[:30]
                    })
                    df["Confidence %"] = (df["Confidence"] * 100).round(1)
                    df["Level"] = df["Confidence"].apply(
                        lambda x: "High" if x >= 0.8 else "Medium" if x >= 0.5 else "Low"
                    )
                    st.dataframe(df[["Word", "Confidence %", "Level"]], use_container_width=True)
            else:
                st.warning("Could not generate report. Try a different image.")
        else:
            st.info("Upload an X-ray image to generate a report")

    st.markdown("---")
    st.markdown("Chest X-Ray Report Generator v1.0 | TensorFlow + Streamlit")
else:
    st.error("Could not load model. Please check that model files are present.")
    st.info("Required: models/best_model_full_dataset.h5 and models/vocab.pkl")
