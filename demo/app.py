"""
ReportCXR Demo — MedGemma 4B-it + QLoRA weighted_v4 adapter
HuggingFace Spaces (ZeroGPU / A10G)
"""

import re
import spaces
import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import AutoModelForImageTextToText
from peft import PeftModel

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "google/medgemma-4b-it"
ADAPTER_ID    = "diegoi-io-0306/reportcxr-medgemma-weighted-v4"

SYSTEM_PROMPT = (
    "You are an expert radiologist. "
    "Write only the Findings section of a radiology report for the chest X-ray shown. "
    "Be concise and clinical. Do not include an Impression section."
)

REPORT_INSTRUCTION = (
    "Generate a structured radiology Findings section for this chest X-ray. "
    "Describe only what you observe in the image."
)

# ── Model loading (once per worker) ───────────────────────────────────────────

processor = None
model = None

def _load_model():
    global processor, model
    if model is not None:
        return

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_ID)
    model.eval()


# ── Inference helpers ──────────────────────────────────────────────────────────

def _run(messages: list, image: Image.Image | None, max_new_tokens: int = 300) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if image is not None:
        # inject pixel values for the <image> token
        img_inputs = processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = img_inputs["pixel_values"]

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


@spaces.GPU
def generate_report(image: Image.Image, indication: str) -> str:
    _load_model()

    if image is None:
        return "Please upload a chest X-ray image first."

    indication = indication.strip() if indication else ""
    user_text = f"{SYSTEM_PROMPT}\nIndication: {indication}" if indication else SYSTEM_PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ],
        }
    ]
    return _run(messages, image, max_new_tokens=350)


@spaces.GPU
def free_chat(image: Image.Image, history: list, user_message: str) -> tuple[list, str]:
    _load_model()

    if not user_message.strip():
        return history, ""

    if image is None:
        history.append({"role": "assistant", "content": "Please upload an X-ray image to start the conversation."})
        return history, ""

    # Build full message list from history
    messages = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Current user turn (always includes the image for context)
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": user_message.strip()},
        ],
    })

    reply = _run(messages, image, max_new_tokens=400)

    history.append({"role": "user",      "content": user_message.strip()})
    history.append({"role": "assistant", "content": reply})
    return history, ""


# ── UI ─────────────────────────────────────────────────────────────────────────

_DESCRIPTION = """
## ReportCXR — Chest X-Ray Report Generation

**Model:** MedGemma 4B-it fine-tuned with QLoRA on the IU X-Ray dataset
**Adapter:** [`diegoi-io-0306/reportcxr-medgemma-weighted-v4`](https://huggingface.co/diegoi-io-0306/reportcxr-medgemma-weighted-v4)

> ⚠️ **Research demo only.** Not for clinical use.
> MedGemma requires accepting [Google's Health AI Developer Foundations license](https://huggingface.co/google/medgemma-4b-it).

Upload a chest X-ray and either **generate a structured Findings report** or **ask free-form questions** about the image.
"""

_REPORT_NOTE = (
    "Uses the trained prompt format: SYSTEM_PROMPT → Indication → Findings. "
    "Leave Indication blank if unknown."
)

with gr.Blocks(title="ReportCXR Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="Chest X-Ray",
                type="pil",
                height=380,
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                # ── Tab 1: Structured report ──────────────────────────────────
                with gr.TabItem("Structured Report"):
                    gr.Markdown(f"*{_REPORT_NOTE}*")
                    indication_box = gr.Textbox(
                        label="Clinical Indication (optional)",
                        placeholder="e.g. Shortness of breath, rule out pneumonia",
                        lines=2,
                    )
                    report_btn = gr.Button("Generate Report", variant="primary")
                    report_out = gr.Textbox(
                        label="Generated Findings",
                        lines=10,
                        show_copy_button=True,
                    )
                    report_btn.click(
                        fn=generate_report,
                        inputs=[img_input, indication_box],
                        outputs=report_out,
                    )

                # ── Tab 2: Free chat ──────────────────────────────────────────
                with gr.TabItem("Free Chat"):
                    gr.Markdown(
                        "Ask any question about the X-ray. "
                        "The model keeps context within the session."
                    )
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        type="messages",
                        height=320,
                    )
                    chat_state = gr.State([])
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="Your question",
                            placeholder="Is there evidence of cardiomegaly?",
                            scale=4,
                            show_label=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    def _submit(image, history, msg):
                        history, _ = free_chat(image, history, msg)
                        return history, history, ""

                    send_btn.click(
                        fn=_submit,
                        inputs=[img_input, chat_state, chat_input],
                        outputs=[chatbot, chat_state, chat_input],
                    )
                    chat_input.submit(
                        fn=_submit,
                        inputs=[img_input, chat_state, chat_input],
                        outputs=[chatbot, chat_state, chat_input],
                    )

                    clear_btn = gr.Button("Clear conversation", size="sm")
                    clear_btn.click(
                        fn=lambda: ([], []),
                        outputs=[chatbot, chat_state],
                    )

    gr.Markdown(
        "**Citation:** IU X-Ray dataset (Demner-Fushman et al., 2016) · "
        "MedGemma (Google DeepMind, 2024) · QLoRA (Dettmers et al., 2023)"
    )


if __name__ == "__main__":
    demo.launch()
