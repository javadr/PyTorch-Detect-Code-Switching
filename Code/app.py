#!/usr/bin/env python3

import streamlit as st
import torch

from config import CFG
from data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# To make a reproducible output
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)


# Load your model
@st.cache_resource  # Cache the model loading for efficiency
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    return model


# Function to predict code-switching
def predict_code_switching(text):
    """Predict if the input text exhibits code-switching behavior."""
    # Preprocess the text (if needed)
    # This is a placeholder. Replace with your actual preprocessing logic.
    tokens = text.split()
    x = Data.embedding_s(Data.chr2id, [tokens])
    out = model(torch.LongTensor(x).to(device)).argmax(dim=-1)[0].tolist()
    labels = [Data.id2lbl[i] for i in out]
    return tokens, labels[:]


# Map language tags to colors
LANGUAGE_COLORS = {
    "en": "#FFD700",  # Gold
    "es": "#FF4500",  # Orange Red
    "other": "#87CEEB",  # Sky Blue
}


# Function to generate colored HTML for tokens
def colorize_tokens(tokens, labels):
    """Create a string of tokens wrapped in HTML spans with colors based on labels."""
    colored_text = ""
    for token, label in zip(tokens, labels):
        color = LANGUAGE_COLORS.get(label, "#FFFFFF")  # Default to white if label not found
        colored_text += f'<span style="background-color: {color}; padding: 3px; margin: 2px; border-radius: 2px;">{token}</span> '
    return colored_text


# Streamlit App Layout
def main():
    # Title and Description
    st.title("Code-Switching Detection")
    st.markdown("""
    This application uses a Character Based CNN+BiLSTM model for Code-Switching prediction in text. 
    Enter a sentence or paragraph below to see the result.
    - **English**: Gold
    - **Spanish**: Orange Red
    - **Other**: Sky Blue
    """)

    # Input Text
    user_input = st.text_area("Enter text to analyze:", height=100)

    # Button to Predict
    if st.button("Analyze"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                # Predict
                tokens, labels = predict_code_switching(user_input)

                # Generate Colored Text
                colored_text = colorize_tokens(tokens, labels)

                # Display Results
                st.subheader("Results:")
                st.markdown(colored_text, unsafe_allow_html=True)  # Render the HTML

        else:
            st.warning("Please enter text before clicking analyze.")


# Run the app
if __name__ == "__main__":
    # Ensure model path is correct
    model_path = CFG.saved_models_path / "bestmodel.pth"
    if model_path.exists():
        model = torch.load(model_path, map_location=torch.device(device))
        main()
    else:
        st.error(f"Model file not found at {model_path}. Please check the path and try again.")
