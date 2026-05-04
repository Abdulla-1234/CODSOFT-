"""
Task 3: Image Captioning using BLIP
CodSoft AI Internship
Author: Doodakula Mohammad Abdulla

Model  : Salesforce/blip-image-captioning-large
         (Vision Transformer encoder + BERT-style language model)

Install:
    pip install -r requirements.txt

Usage:
    python image_captioning.py caption path/to/photo.jpg
    python image_captioning.py caption https://example.com/image.jpg
    python image_captioning.py caption photo.jpg --prompt "a photo of"
    python image_captioning.py ui                     # Gradio web demo
    python image_captioning.py demo                   # quick test with sample URL
"""

import sys
import argparse
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


# ─── Model ────────────────────────────────────────────────────────────────────

MODEL_ID = "Salesforce/blip-image-captioning-large"
_processor = None
_model     = None
_device    = None

def load_model():
    global _processor, _model, _device
    if _model is not None:
        return _processor, _model, _device
    print("⏳ Loading BLIP model (first run downloads ~1.9 GB)...")
    _device    = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = BlipProcessor.from_pretrained(MODEL_ID)
    _model     = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(_device)
    print(f"✅ Model loaded on {_device.upper()}.\n")
    return _processor, _model, _device


# ─── Caption ──────────────────────────────────────────────────────────────────

def generate_caption(image: Image.Image,
                     processor, model, device,
                     conditional_text: str = None,
                     max_new_tokens: int = 50) -> str:
    image = image.convert("RGB")
    if conditional_text:
        inputs = processor(image, conditional_text, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(ids[0], skip_special_tokens=True)


# ─── Image Loader ─────────────────────────────────────────────────────────────

def load_image(source: str) -> Image.Image:
    """Load from local path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"  🌐 Fetching image from URL...")
        try:
            r = requests.get(source, stream=True, timeout=15)
            r.raise_for_status()
            return Image.open(BytesIO(r.content))
        except Exception as e:
            raise RuntimeError(f"Failed to download image: {e}")
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: '{source}'\n"
                f"  → Make sure to provide the FULL path, e.g.:\n"
                f"     python image_captioning.py caption C:\\Users\\You\\Pictures\\photo.jpg\n"
                f"  → Or use a URL:\n"
                f"     python image_captioning.py caption https://example.com/image.jpg"
            )
        return Image.open(path)


# ─── Modes ────────────────────────────────────────────────────────────────────

def mode_caption(args):
    processor, model, device = load_model()
    for src in args.images:
        print(f"🖼  Processing: {src}")
        try:
            image = load_image(src)
        except Exception as e:
            print(f"  ❌ {e}\n")
            continue
        caption = generate_caption(image, processor, model, device,
                                   conditional_text=args.prompt,
                                   max_new_tokens=args.max_tokens)
        print(f"  📝 Caption: {caption}\n")


def mode_demo(args):
    """Quick demo using a royalty-free sample image."""
    SAMPLE_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/"
        "Cat03.jpg/320px-Cat03.jpg"
    )
    processor, model, device = load_model()
    print(f"🎬 Running demo with sample image (cat)...")
    print(f"   URL: {SAMPLE_URL}\n")
    try:
        image = load_image(SAMPLE_URL)
    except Exception as e:
        print(f"❌ Could not load sample image: {e}")
        return
    caption = generate_caption(image, processor, model, device)
    print(f"📝 Caption: {caption}\n")
    print("✅ Demo complete! Now try with your own image:")
    print("   python image_captioning.py caption YOUR_IMAGE.jpg")


def mode_ui(args):
    try:
        import gradio as gr
    except ImportError:
        sys.exit("❌ Gradio not installed. Run: pip install gradio")

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
            gr.Textbox(label="Conditional Prompt (optional)",
                       placeholder='e.g. "a photo of"'),
            gr.Slider(10, 100, value=50, step=5, label="Max Tokens"),
        ],
        outputs=gr.Textbox(label="Generated Caption"),
        title="🖼 Image Captioning — CodSoft AI Internship Task 3",
        description=(
            "Upload any image to get an AI-generated caption using **Salesforce BLIP**.\n"
            "Model: blip-image-captioning-large (Vision Transformer + BERT)"
        ),
        theme="soft",
    )
    print("🚀 Launching Gradio UI... (open the URL shown below)")
    demo.launch(share=False)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 57)
    print("  🤖  Image Captioning (BLIP)  |  CodSoft AI Task 3")
    print("=" * 57 + "\n")

    parser = argparse.ArgumentParser(
        description="Image Captioning using BLIP | CodSoft AI Internship"
    )
    sub = parser.add_subparsers(dest="mode")

    # caption
    p = sub.add_parser("caption", help="Caption image(s) from a local path or URL")
    p.add_argument("images", nargs="+",
                   help="Local file path(s) or URL(s)")
    p.add_argument("--prompt", default=None,
                   help='Optional prefix, e.g. "a photo of"')
    p.add_argument("--max-tokens", type=int, default=50,
                   help="Max caption tokens (default: 50)")

    # demo
    sub.add_parser("demo", help="Run a quick demo with a sample online image")

    # ui
    sub.add_parser("ui", help="Launch Gradio web UI")

    args = parser.parse_args()

    if   args.mode == "caption": mode_caption(args)
    elif args.mode == "demo":    mode_demo(args)
    elif args.mode == "ui":      mode_ui(args)
    else:
        print("Usage examples:\n")
        print("  # Caption a local image (use full path on Windows):")
        print("  python image_captioning.py caption C:\\Users\\You\\photo.jpg\n")
        print("  # Caption from a URL:")
        print("  python image_captioning.py caption https://example.com/img.jpg\n")
        print("  # Quick demo with a sample image (no file needed):")
        print("  python image_captioning.py demo\n")
        print("  # Launch Gradio web UI:")
        print("  python image_captioning.py ui\n")


if __name__ == "__main__":
    main()
