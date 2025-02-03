import base64
import io
from typing import Tuple
from PIL import Image

def detect_image(base64_string: str, max_size: int = 1048576) -> Tuple[str, str, int]:
    try:
        # Decode Base64 string
        image_data = base64.b64decode(base64_string)
        image_stream = io.BytesIO(image_data)

        # Open the image
        with Image.open(image_stream) as img:
            # Convert to RGB if not in RGB mode (required for JPEG)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Initialize compression quality
            quality = 95
            output_stream = io.BytesIO()

            # Adjust quality until size is under max_size (1MB)
            while quality > 10:
                output_stream.seek(0)
                output_stream.truncate()
                img.save(output_stream, format="JPEG", quality=quality)

                # Check size
                if output_stream.tell() <= max_size:
                    break
                quality -= 5  # Reduce quality to lower size
            
            # Encode back to Base64
            output_stream.seek(0)
            compressed_base64 = base64.b64encode(output_stream.read()).decode("utf-8")

            return "image/jpeg", compressed_base64, quality

    except Exception as e:
        return {"error": str(e)}

__all__ = ['detect_image']