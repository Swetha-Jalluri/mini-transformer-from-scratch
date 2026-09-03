from pathlib import Path

import streamlit as st
import torch

from src.model import MiniTransformer


st.set_page_config(
    page_title="Mini Transformer",
    page_icon="🧠",
    layout="centered"
)

project_root = Path(__file__).parent
checkpoint_path = (
    project_root / "checkpoints" / "mini_transformer.pt"
)


@st.cache_resource
def load_model():
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model = MiniTransformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return (
        model,
        checkpoint["char_to_id"],
        checkpoint["id_to_char"]
    )


model, char_to_id, id_to_char = load_model()

st.title("Mini Transformer Text Generator")

st.write(
    "Enter some starting text and the model will continue it "
    "one character at a time."
)

prompt = st.text_area(
    "Starting text",
    value="ROMEO:",
    height=100
)

length = st.slider(
    "Characters to generate",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)

temperature = st.slider(
    "Creativity",
    min_value=0.2,
    max_value=1.5,
    value=0.8,
    step=0.1
)

if st.button("Generate", type="primary"):
    if not prompt:
        prompt = "\n"

    unsupported = sorted(set(prompt) - set(char_to_id))

    if unsupported:
        st.error(f"Unsupported characters: {unsupported}")

    else:
        input_tokens = torch.tensor(
            [[char_to_id[character] for character in prompt]],
            dtype=torch.long
        )

        with st.spinner("Generating text..."):
            output_tokens = model.generate(
                input_tokens,
                max_new_tokens=length,
                temperature=temperature
            )

        generated_text = "".join(
            id_to_char[token]
            for token in output_tokens[0].tolist()
        )

        st.text_area(
            "Generated text",
            value=generated_text,
            height=350
        )