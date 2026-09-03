from pathlib import Path

import gradio as gr
import torch

from src.model import MiniTransformer


project_root = Path(__file__).parent
checkpoint_file = (
    project_root / "checkpoints" / "mini_transformer.pt"
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

checkpoint = torch.load(
    checkpoint_file,
    map_location=device,
    weights_only=False
)

config = checkpoint["config"]
char_to_id = checkpoint["char_to_id"]
id_to_char = checkpoint["id_to_char"]

model = MiniTransformer(**config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def generate_text(prompt, length, temperature):
    if not prompt:
        prompt = "\n"

    unsupported = set(prompt) - set(char_to_id)

    if unsupported:
        return f"Unsupported characters: {unsupported}"

    input_tokens = torch.tensor(
        [[char_to_id[character] for character in prompt]],
        dtype=torch.long,
        device=device
    )

    output_tokens = model.generate(
        input_tokens,
        max_new_tokens=int(length),
        temperature=float(temperature)
    )

    return "".join(
        id_to_char[token]
        for token in output_tokens[0].cpu().tolist()
    )


with gr.Blocks() as demo:
    gr.Markdown(
        "# Mini Transformer Text Generator\n"
        "A character-level Transformer built from scratch in PyTorch."
    )

    prompt = gr.Textbox(
        label="Starting text",
        value="ROMEO:"
    )

    length = gr.Slider(
        minimum=50,
        maximum=500,
        value=200,
        step=50,
        label="Characters to generate"
    )

    temperature = gr.Slider(
        minimum=0.2,
        maximum=1.5,
        value=0.8,
        step=0.1,
        label="Creativity"
    )

    generate_button = gr.Button("Generate")
    output = gr.Textbox(label="Generated text", lines=15)

    generate_button.click(
        generate_text,
        inputs=[prompt, length, temperature],
        outputs=output
    )


if __name__ == "__main__":
    demo.launch()