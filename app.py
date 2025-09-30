from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import time
import base64
import requests
from datetime import datetime
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

load_dotenv()


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def enrich_prompt(prompt: str) -> str:
    """Refine the user's prompt using GROQ first (if configured), then OpenAI; otherwise pass-through."""
    # Try Groq (Llama) first
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            system = (
                "You are a prompt engineer for interior design image generation. "
                "Rewrite the user's prompt into a descriptive, visual, style-rich prompt suitable "
                "for Stable Diffusion. Keep it under 120 words."
            )
            resp = client.chat.completions.create(
                model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=200,
            )
            refined = resp.choices[0].message.content.strip()
            if refined:
                return refined
        except Exception:
            pass

    # Fallback to OpenAI if available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            system = (
                "You are a prompt engineer for interior design image generation. "
                "Rewrite the user's prompt into a descriptive, visual, style-rich prompt suitable "
                "for Stable Diffusion. Keep it under 120 words."
            )
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=200,
            )
            refined = resp.choices[0].message.content.strip()
            if refined:
                return refined
        except Exception:
            pass

    return prompt


def generate_with_stability(prompt: str, num_images: int = 2, steps: int = 30):
    """Call Stability AI REST API to generate images for a given prompt."""
    stability_api_key = os.environ.get("STABILITY_API_KEY")
    # Prefer SDXL by default; users can override via STABILITY_MODEL
    model = os.environ.get("STABILITY_MODEL", "stable-diffusion-xl-1024-v1-0")
    if not stability_api_key:
        raise RuntimeError("STABILITY_API_KEY not set in environment")

    endpoint = f"https://api.stability.ai/v1/generation/{model}/text-to-image"

    # SDXL requires specific dimensions; default to 1024x1024 for SDXL engines
    if "xl" in model:
        height = 1024
        width = 1024
    else:
        height = 512
        width = 512

    payload = {
        "text_prompts": [
            {"text": prompt}
        ],
        "cfg_scale": 7,
        "clip_guidance_preset": "FAST_BLUE",
        "height": height,
        "width": width,
        "samples": max(1, min(num_images, 4)),
        "steps": steps,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {stability_api_key}",
    }

    r = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    if r.status_code == 404:
        # Retry with a known engine if the provided engine isn't found
        fallback_engines = [
            "stable-diffusion-xl-1024-v1-0",
            "stable-diffusion-v1-6",
        ]
        for eng in fallback_engines:
            if eng == model:
                continue
            r2 = requests.post(
                f"https://api.stability.ai/v1/generation/{eng}/text-to-image",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if r2.status_code == 200:
                r = r2
                break
        if r.status_code != 200:
            raise RuntimeError(f"Stability API error: {r.status_code} - {r.text[:200]}")
    elif r.status_code != 200:
        raise RuntimeError(f"Stability API error: {r.status_code} - {r.text[:200]}")

    data = r.json()
    artifacts = data.get("artifacts", [])
    if not artifacts:
        raise RuntimeError("No images returned by Stability API")

    out_paths = []
    os.makedirs("Results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    for idx, art in enumerate(artifacts):
        b64 = art.get("base64")
        if not b64:
            continue
        image_bytes = base64.b64decode(b64)
        filename = f"{ts}_{idx:02d}.png"
        out_path = os.path.join("Results", filename)
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        out_paths.append(out_path)

    return out_paths


@app.route('/explore', methods=['GET', 'POST'])
def explore():
    error = None
    generated = []
    user_prompt = ''

    if request.method == 'POST':
        user_prompt = (request.form.get('prompt') or '').strip()
        num_images = int(request.form.get('num_images') or 2)
        if not user_prompt:
            error = 'Please enter a prompt.'
        else:
            try:
                refined = enrich_prompt(user_prompt)
                generated = generate_with_stability(refined, num_images=num_images)
            except Exception as e:
                error = str(e)

    # List existing results for gallery view
    gallery = []
    try:
        for name in sorted(os.listdir('Results')):
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                gallery.append(os.path.join('Results', name).replace('\\', '/'))
    except FileNotFoundError:
        pass

    return render_template('explore.html', error=error, generated=generated, gallery=gallery, prompt_value=user_prompt)


@app.route('/Results/<path:filename>')
def serve_results(filename: str):
    """Serve generated and sample images from the Results directory."""
    return send_from_directory('Results', filename)


@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """Transcribe uploaded audio into text for prompt input.
    Prefers GROQ Whisper, falls back to OpenAI Whisper if available.
    """
    if 'audio' not in request.files:
        return {"error": "No audio file provided"}, 400
    audio = request.files['audio']
    if audio.filename == '':
        return {"error": "Empty filename"}, 400

    temp_dir = 'Results'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"transcribe_{int(time.time())}.wav")
    audio.save(temp_path)

    text = None
    # Try Groq Whisper first
    groq_key = os.environ.get('GROQ_API_KEY')
    if groq_key and text is None:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            with open(temp_path, 'rb') as f:
                resp = client.audio.transcriptions.create(
                    file=(os.path.basename(temp_path), f, 'audio/wav'),
                    model=os.environ.get('GROQ_WHISPER_MODEL', 'whisper-large-v3'),
                )
            text = (resp.text or '').strip()
        except Exception:
            text = None

    # Fallback to OpenAI Whisper if configured
    if text is None:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                with open(temp_path, 'rb') as f:
                    resp = client.audio.transcriptions.create(
                        model=os.environ.get('OPENAI_WHISPER_MODEL', 'whisper-1'),
                        file=f,
                    )
                text = (resp.text or '').strip()
            except Exception:
                text = None

    try:
        os.remove(temp_path)
    except Exception:
        pass

    if not text:
        return {"error": "Transcription failed"}, 500
    return {"text": text}


@app.route('/api/flag', methods=['POST'])
def api_flag():
    """Placeholder endpoint to receive flag actions from UI."""
    img = request.form.get('image')
    reason = request.form.get('reason', '')
    # For now, simply log to server console
    print(f"Flagged image: {img} reason: {reason}")
    return {"ok": True}


if __name__ == "__main__":
    app.run(debug=True, port=2000)
