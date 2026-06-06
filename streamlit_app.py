import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Chest X-Ray Report Generator",
    page_icon="🩻",
    layout="wide"
)

st.markdown("# Chest X-Ray Automated Report Generator")
st.markdown("AI-powered medical report generation using **CheXNet + BioGPT**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("About This Project")
    st.info("""
    **Architecture**: 
    - CheXNet (DenseNet121 for chest X-ray features)
    - LLaVA-style image prefix
    - BioGPT (medical text decoder)
    
    **Training Data**: IU-Xray dataset (7,470 images)
    
    **Test Metrics**:
    - BLEU: 0.062
    - ROUGE-1: 0.328
    - ROUGE-L: 0.184
    - Diversity: 93.88%
    """)
    st.warning("⚠️ Research/educational use only. NOT for clinical decisions.")
    st.markdown("---")
    st.caption("Model: Sudharsan66/chest-xray-chexnet-biogpt")


# ============================================================
# ARCHITECTURE (must match training)
# ============================================================

class CheXNetFeatures(nn.Module):
    def __init__(self, chexnet_model):
        super().__init__()
        self.features = chexnet_model.features
    
    def forward(self, x):
        features = self.features(x)
        batch_size, channels, h, w = features.shape
        features = features.view(batch_size, channels, h*w).permute(0, 2, 1)
        return features


class ChestXrayReportModel(nn.Module):
    def __init__(self, image_encoder, biogpt_model, tokenizer, num_image_tokens=49):
        super().__init__()
        self.image_encoder = image_encoder
        self.biogpt = biogpt_model
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        
        chexnet_dim = 1024
        biogpt_dim = biogpt_model.config.hidden_size
        
        self.image_projection = nn.Sequential(
            nn.Linear(chexnet_dim, biogpt_dim * 2),
            nn.GELU(),
            nn.Linear(biogpt_dim * 2, biogpt_dim),
            nn.LayerNorm(biogpt_dim)
        )
    
    def get_image_embeds(self, pixel_values):
        with torch.no_grad():
            image_features = self.image_encoder(pixel_values)
        image_embeds = self.image_projection(image_features)
        return image_embeds
    
    @torch.no_grad()
    def generate(self, pixel_values, max_length=120, repetition_penalty=1.5,
                 no_repeat_ngram_size=3, min_length=15):
        self.eval()
        device = pixel_values.device
        batch_size = pixel_values.size(0)
        
        image_embeds = self.get_image_embeds(pixel_values)
        
        input_ids = torch.full(
            (batch_size, 1), self.tokenizer.bos_token_id,
            dtype=torch.long, device=device
        )
        
        for _ in range(max_length - 1):
            text_embeds = self.biogpt.biogpt.embed_tokens(input_ids)
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            
            attention_mask = torch.ones(
                batch_size, self.num_image_tokens + input_ids.size(1),
                dtype=torch.long, device=device
            )
            
            outputs = self.biogpt.biogpt(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            text_hidden = outputs.last_hidden_state[:, self.num_image_tokens:, :]
            logits = self.biogpt.output_projection(text_hidden)
            next_logits = logits[:, -1, :].clone()
            
            for b in range(batch_size):
                for tok in set(input_ids[b].tolist()):
                    next_logits[b, tok] /= repetition_penalty
            
            if no_repeat_ngram_size > 0 and input_ids.size(1) >= no_repeat_ngram_size:
                for b in range(batch_size):
                    seq = input_ids[b].tolist()
                    last_ngram = tuple(seq[-(no_repeat_ngram_size-1):])
                    banned = set()
                    for i in range(len(seq) - no_repeat_ngram_size + 1):
                        ng = tuple(seq[i:i+no_repeat_ngram_size])
                        if ng[:-1] == last_ngram:
                            banned.add(ng[-1])
                    for tok in banned:
                        next_logits[b, tok] = -float('inf')
            
            if input_ids.size(1) < min_length:
                next_logits[:, self.tokenizer.eos_token_id] = -float('inf')
            
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        return input_ids


# ============================================================
# MODEL LOADING (cached)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_components():
    """Load all model components - cached so it runs only once"""
    from transformers import BioGptTokenizer, BioGptForCausalLM
    import torchxrayvision as xrv
    from huggingface_hub import hf_hub_download
    
    device = torch.device("cpu")
    
    # Download our trained weights from HF Hub
    weights_path = hf_hub_download(
        repo_id="Sudharsan66/chest-xray-chexnet-biogpt",
        filename="best_model.pt"
    )
    
    # Load BioGPT
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    biogpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    biogpt.resize_token_embeddings(len(tokenizer))
    
    # Load CheXNet
    chexnet = xrv.models.DenseNet(weights="densenet121-res224-all")
    chexnet.eval()
    
    image_encoder = CheXNetFeatures(chexnet).to(device)
    image_encoder.eval()
    for p in image_encoder.parameters():
        p.requires_grad = False
    
    # Build full model
    model = ChestXrayReportModel(image_encoder, biogpt, tokenizer)
    
    # Load trained weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================

def preprocess_image(image):
    """Match training preprocessing exactly"""
    img = image.convert('L')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - 127.5) / 127.5 * 1024
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor


def generate_report(image, model, tokenizer):
    pixel_values = preprocess_image(image)
    
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=120,
            min_length=15,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3
        )
    
    report = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return report.strip()


# ============================================================
# MAIN APP
# ============================================================

try:
    with st.spinner("Loading model (first time takes 2-3 minutes)..."):
        model, tokenizer = load_model_components()
    st.success("Model loaded! Ready to generate reports.")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.info("If you see memory errors, the app may need more resources. Contact support.")
    st.stop()

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload X-Ray Image")
    st.caption("Supported formats: JPG, JPEG, PNG")
    
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        
        st.write("**Image Details:**")
        st.write(f"- Size: {image.size}")
        st.write(f"- Mode: {image.mode}")

with col2:
    st.subheader("Generated Report")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing X-ray and generating report (~20-40 seconds)..."):
            report = generate_report(image, model, tokenizer)
        
        st.markdown("**Medical Report:**")
        st.info(report)
        
        st.markdown("---")
        with st.expander("About this generation"):
            st.write("""
            - **Image features**: Extracted by CheXNet (trained on 14 chest pathologies)
            - **Text generation**: BioGPT trained on 15M medical abstracts
            - **Architecture**: LLaVA-style image prefix tuning
            - **Diversity**: This model generates unique reports (93.88% unique on test set)
            """)
    else:
        st.info("Upload an X-ray image to generate a report")

st.markdown("---")
st.caption("Chest X-Ray Report Generator v2.0 | CheXNet + BioGPT | Built with PyTorch + Streamlit")
