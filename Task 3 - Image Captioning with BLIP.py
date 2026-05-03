"""
Task 3: Image Captioning
CodSoft AI Internship
Author: Doodakula Mohammad Abdulla

Uses Salesforce BLIP (Bootstrapping Language-Image Pre-training) —
a state-of-the-art vision-language model that combines a ViT image encoder
with a BERT-style language model for caption generation.

Install:
    pip install torch torchvision transformers pillow requests gradio
"""

import sys
import argparse
from pathlib import Path

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


# ─── Model Loader ─────────────────────────────────────────────────────────────

MODEL_ID = "Salesforce/blip-image-captioning-large"

def load_model():
    print("⏳ Loading BLIP model (first run may take a minute)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model     = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    print(f"✅ Model loaded on {device.upper()}.\n")
    return processor, model, device


# ─── Caption Generator ────────────────────────────────────────────────────────

def generate_caption(image: Image.Image, processor, model, device,
                     conditional_text: str = None,
                     max_new_tokens: int = 50) -> str:
    """
    Generate a caption for a PIL Image.

    Args:
        image:            PIL.Image (RGB)
        conditional_text: Optional prompt prefix, e.g. "a photo of"
        max_new_tokens:   Max tokens for the generated caption
    Returns:
        Caption string
    """
    image = image.convert("RGB")

    if conditional_text:
        inputs = processor(image, conditional_text, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption


# ─── Image Loader ─────────────────────────────────────────────────────────────

def load_image(source: str) -> Image.Image:
    """Load image from a local path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source, stream=True, timeout=10)
        response.raise_for_status()
        from io import BytesIO
        return Image.open(BytesIO(response.content))
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {source}")
    return Image.open(path)


# ─── CLI Mode ─────────────────────────────────────────────────────────────────

def cli_mode(args):
    processor, model, device = load_model()
    sources = args.images

    for src in sources:
        print(f"🖼  Processing: {src}")
        try:
            image = load_image(src)
        except Exception as e:
            print(f"  ❌ Could not load image: {e}\n")
            continue

        caption = generate_caption(image, processor, model, device,
                                   conditional_text=args.prompt,
                                   max_new_tokens=args.max_tokens)
        print(f"  📝 Caption: {caption}\n")


# ─── Gradio Web UI ────────────────────────────────────────────────────────────

def launch_gradio():
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    processor, model, device = load_model()

    def caption_image(image, prompt, max_tokens):
        if image is None:
            return "Please upload an image."
        pil_img = Image.fromarray(image)
        return generate_caption(pil_img, processor, model, device,
                                conditional_text=prompt or None,
                                max_new_tokens=int(max_tokens))

    demo = gr.Interface(
        fn=caption_image,
        inputs=[
            gr.Image(label="Upload Image"),
            gr.Textbox(label="Conditional Prompt (optional)", placeholder="a photo of"),
            gr.Slider(10, 100, value=50, step=5, label="Max Tokens"),
        ],
        outputs=gr.Textbox(label="Generated Caption"),
        title="🖼 Image Captioning — CodSoft AI Internship",
        description="Upload any image to get an AI-generated caption using the BLIP model.",
        examples=[],
        theme="soft",
    )
    demo.launch(share=False)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Captioning using BLIP | CodSoft AI Task 3"
    )
    subparsers = parser.add_subparsers(dest="mode")

    # CLI sub-command
    cli_parser = subparsers.add_parser("caption", help="Caption image(s) from CLI")
    cli_parser.add_argument("images", nargs="+", help="Local paths or URLs")
    cli_parser.add_argument("--prompt", default=None,
                            help='Conditional text prompt, e.g. "a photo of"')
    cli_parser.add_argument("--max-tokens", type=int, default=50,
                            help="Maximum caption tokens (default: 50)")

    # Gradio sub-command
    subparsers.add_parser("ui", help="Launch Gradio web UI")

    return parser.parse_args()


def main():
    print("=" * 55)
    print("  🤖  Image Captioning (BLIP)  |  CodSoft AI Task 3")
    print("=" * 55 + "\n")

    args = parse_args()

    if args.mode == "caption":
        cli_mode(args)
    elif args.mode == "ui":
        launch_gradio()
    else:
        print("Usage examples:")
        print("  # Caption a local image:")
        print("  python image_captioning.py caption photo.jpg")
        print()
        print("  # Caption from a URL:")
        print("  python image_captioning.py caption https://example.com/img.jpg")
        print()
        print("  # Caption with a conditional prompt:")
        print('  python image_captioning.py caption photo.jpg --prompt "a photo of"')
        print()
        print("  # Launch Gradio web UI:")
        print("  python image_captioning.py ui")


if __name__ == "__main__":
    main()
