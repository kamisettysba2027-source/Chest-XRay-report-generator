import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout,
    Add, RepeatVector, TimeDistributed, Lambda
)
from tensorflow.keras.applications import InceptionV3
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
    st.info("AI-Powered Medical Imaging System using CNN-LSTM. Trained on 7470 images.")
    st.markdown("---")
    st.warning("DISCLAIMER: For research/educational purposes only. Not for medical diagnosis.")
    st.markdown("---")
    st.caption("Built with TensorFlow and Streamlit")


def build_model(vocab_size, max_seq_length=150, embedding_dim=256, lstm_units=256):
    """Rebuild the model architecture (same as training)"""
    
    tf.keras.backend.clear_session()
    
    # Image encoder
    inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    inception.trainable = False
    
    image_input = Input(shape=(256, 256, 3), name='image_input')
    features = inception(image_input)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    image_features = Dense(embedding_dim, activation='relu', name='image_features')(features)
    image_features_expanded = RepeatVector(max_seq_length, name='repeat_image')(image_features)
    
    # Text decoder
    text_input = Input(shape=(max_seq_length,), name='text_input')
    embedding = Embedding(vocab_size, embedding_dim, name='embedding')(text_input)
    lstm1 = LSTM(lstm_units, return_sequences=True, dropout=0.5, name='lstm1')(embedding)
    lstm2 = LSTM(lstm_units, return_sequences=True, dropout=0.5, name='lstm2')(lstm1)
    
    # Merge
    merged = Add(name='merge')([image_features_expanded, lstm2])
    merged = Dropout(0.5, name='dropout1')(merged)
    dense1 = TimeDistributed(Dense(512, activation='relu'), name='dense1')(merged)
    dense1 = Dropout(0.5, name='dropout2')(dense1)
    output = TimeDistributed(Dense(vocab_size, activation='softmax'), name='output')(dense1)
    output = Lambda(lambda x: x[:, :-1, :], name='slice_output')(output)
    
    model = Model(inputs=[image_input, text_input], outputs=output)
    return model


@st.cache_resource
def load_model_and_vocab():
    try:
        # Load vocabulary first to know vocab_size
        vocab_path = "models/vocab.pkl"
        if not os.path.exists(vocab_path):
            st.error("Vocabulary file (vocab.pkl) not found")
            return None, None
        
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        
        st.info(f"Loaded vocabulary: {len(vocab)} words")
        
        # Build model architecture
        st.info("Building model architecture...")
        model = build_model(vocab_size=len(vocab))
        
        # Try loading weights from available files
        weight_paths = [
            "models/best_model_full_dataset.weights.h5",
            "models/best_model_full_dataset.h5",
            "models/best_model_2500_gen.h5",
            "models/best_model_1000_gen.h5",
        ]
        
        loaded = False
        for path in weight_paths:
            if os.path.exists(path):
                try:
                    st.info(f"Trying to load weights from: {path}")
                    model.load_weights(path)
                    st.success(f"Loaded weights from: {path}")
                    loaded = True
                    break
                except Exception as e:
                    st.warning(f"Could not load {path}: {str(e)[:100]}")
                    continue
        
        if not loaded:
            st.error("Could not load any model weights")
            return None, None
        
        return model, vocab
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None


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
