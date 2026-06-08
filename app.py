import gradio as gr
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load TinyLlama
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
)

# Load your LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "."
)

def generate_response(question):
    prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    full_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "### Response:" in full_text:
        return full_text.split("### Response:")[-1].strip()

    return full_text

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a question..."
    ),
    outputs="text",
    title="TinyLlama AI Engineering Assistant",
    description="Fine-tuned TinyLlama using LoRA on AI Engineering concepts."
)

demo.launch()