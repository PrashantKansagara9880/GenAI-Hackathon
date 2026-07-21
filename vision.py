from PIL import Image
import io
import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from safety import safe_call
load_dotenv()
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def image_to_data_uri(filepath:str,scale_percent: int = 100,jpeg_quality: int = 85):
    img = Image.open(filepath).convert("RGB")
    original_width = img.width
    original_height = img.height

    scale_percent = max(10, min(scale_percent, 100))
    jpeg_quality = max(20, min(jpeg_quality, 95))

    new_width = max(1, int(original_width * scale_percent / 100))
    new_height = max(1, int(original_height * scale_percent / 100))

    img = img.resize((new_width, new_height), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True
    )

    image_bytes = buffer.getvalue()

    encoded = base64.b64encode(image_bytes).decode()

    data_uri = f"data:image/jpeg;base64,{encoded}"

    image_size_kb = round(len(image_bytes) / 1024, 2)
    # Very rough approximation
    estimated_tokens = len(encoded) // 4

    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "new_width": new_width,
        "new_height": new_height,
        "resolution_percent": scale_percent,
        "jpeg_quality": jpeg_quality,
        "image_size_kb": image_size_kb,
        "base64_length": len(encoded),
        "estimated_tokens": estimated_tokens,
        "recommended": estimated_tokens < 2000
    }

    return {
        "data_uri": data_uri,
        "metadata": metadata
    }

@safe_call
def ask_vision(image_uri, image_ready,question="Describe the image in detail, including any text,objects, and context."):
    if not image_ready:
        return (
            "Upload image first. If already uploaded,follow the instruction below and try again\n\n"
            "⚠️ The uploaded image is too large for analysis.\n\n"
            "Please reduce the image resolution using the slider "
            "and click 'Process Image' again."
        )
    image_data=image_uri.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    response = client.models.generate_content(
        model="models/gemini-3.5-flash",
        contents=[
            question,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            )
        ]
    )
    return response.text