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
checkpoint_path = project_root / "checkpoints" / "mini_transformer.pt"


@st.cache_resource
def load_model():
    """Load the trained model and character vocabulary once."""

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
    value="ROMEO:"
)

length = st.slider(
    "Minimum characters to generate",
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

st.caption(
    "The model may generate a few extra characters to finish the sentence."
)


if st.button("Generate"):
    if not prompt:
        st.warning("Enter some starting text.")

    else:
        unsupported_characters = sorted(
            set(prompt) - set(char_to_id)
        )

        if unsupported_characters:
            readable_characters = ", ".join(
                repr(character)
                for character in unsupported_characters
            )

            st.error(
                "The model does not recognize these characters: "
                f"{readable_characters}"
            )

        else:
            input_tokens = torch.tensor(
                [[char_to_id[character] for character in prompt]],
                dtype=torch.long
            )

            with st.spinner("Generating text..."):
                output_tokens = model.generate(
                    input_tokens,
                    max_new_tokens=length + 300,
                    temperature=temperature
                )

            generated_text = "".join(
                id_to_char[token]
                for token in output_tokens[0].tolist()
            )

            # Search for a natural ending after the minimum length.
            target_end = len(prompt) + length
            search_start = max(len(prompt), target_end - 1)

            ending = next(
                (
                    index + 1
                    for index in range(
                        search_start,
                        len(generated_text)
                    )
                    if generated_text[index] in ".!?"
                ),
                None
            )

            # If there is no punctuation, stop between words.
            if ending is None:
                ending = generated_text.rfind(
                    " ",
                    target_end
                )

                if ending == -1:
                    ending = len(generated_text)

            generated_text = generated_text[:ending].rstrip()

            st.text_area(
                "Generated text",
                value=generated_text,
                height=350
            )