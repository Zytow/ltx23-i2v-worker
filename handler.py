import runpod
import torch
import base64
import io
import os
import tempfile
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/ltx-video-2.3")
MODEL_ID   = "Lightricks/LTX-Video"

print(f"Lade LTX-Video 2.3 von: {MODEL_PATH}")
pipe = LTXImageToVideoPipeline.from_pretrained(
    MODEL_PATH if os.path.exists(f"{MODEL_PATH}/model_index.json") else MODEL_ID,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.vae.enable_tiling()
print("LTX-Video 2.3 bereit")

def handler(job):
    inp       = job["input"]
    image_b64 = inp.get("image_base64")
    image_url = inp.get("image_url")
    width     = inp.get("width", 1280)
    height    = inp.get("height", 720)
    fps       = inp.get("fps", 16)
    seed      = inp.get("seed", -1)

    if image_b64:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB").resize((width, height))
    elif image_url:
        image = load_image(image_url).resize((width, height))
    else:
        return {"error": "image_base64 oder image_url erforderlich"}

    gen = torch.Generator("cuda")
    if seed != -1:
        gen.manual_seed(seed)

    result = pipe(
        image=image,
        prompt=inp.get("prompt", "smooth cinematic motion, high quality"),
        negative_prompt=inp.get("negative_prompt", "blur, watermark, shaky, distortion"),
        width=width,
        height=height,
        num_frames=inp.get("num_frames", 97),
        num_inference_steps=inp.get("steps", 30),
        guidance_scale=inp.get("guidance_scale", 3.5),
        generator=gen,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name
    export_to_video(result.frames[0], out_path, fps=fps)

    with open(out_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    os.unlink(out_path)

    num_frames = inp.get("num_frames", 97)
    return {
        "video_base64": video_b64,
        "format": "mp4",
        "fps": fps,
        "frames": num_frames,
        "duration_sec": round(num_frames / fps, 2),
        "width": width,
        "height": height,
    }

runpod.serverless.start({"handler": handler})
