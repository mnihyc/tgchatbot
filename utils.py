import base64
import io
from typing import Tuple, List
from PIL import Image
import av
import os

def extract_keyframes(webm_data, num_key_frames=10):
    """
    Decodes the input base64 data of a .webm file (e.g., a Telegram sticker),
    extracts up to `max_frames` *key frames only*, and returns a list of Pillow
    ImageFile objects (such as PIL.PngImagePlugin.PngImageFile).
    
    :param webm_data: Base64-decoded string of the .webm file content.
    :param max_frames: Maximum number of key frames to extract (default 10).
    :return: A list of up to `max_frames` Pillow ImageFile objects.
    """
    #webm_data = base64.b64decode(webm_base64_data)
    container = av.open(io.BytesIO(webm_data))

    stream = container.streams.video[0]
    # Only decode key frames
    stream.skip_frame = "NONKEY"

    # Decode all key frames
    all_keyframes = []
    for frame in container.decode(stream):
        # Convert to PIL
        pil_image = frame.to_image()
        all_keyframes.append((frame.pts, pil_image))
        if len(all_keyframes) >= 100000:
            # Too many frames, abort
            break

    # Sort by presentation timestamp, just in case (PyAV normally does this already).
    all_keyframes.sort(key=lambda x: x[0])

    # Decide which key frames to keep: e.g., evenly sample num_key_frames from them
    if len(all_keyframes) <= num_key_frames:
        chosen_frames = all_keyframes
    else:
        step = len(all_keyframes) / float(num_key_frames)
        chosen_frames = [
            all_keyframes[int(i * step)]
            for i in range(num_key_frames)
        ]
    
    # Convert chosen frames to Pillow ImageFile objects
    out_images = []
    for pts, pil_img in chosen_frames:
        png_buffer = io.BytesIO()
        pil_img.save(png_buffer, format="PNG")
        png_buffer.seek(0)
        img_file = Image.open(png_buffer)
        out_images.append(img_file)

    container.close()
    return out_images


def detect_image(image_data: str, max_size: int = 1048576, split_frames: int = 10) -> List[Tuple[str, str, int]]:
    """
    Open an image (supports formats like PNG, JPEG, WebP, APNG, etc.), 
    extract up to 5 frames if animated, and return base64-encoded image data.
    
    Parameters:
        image_data: Byte content or file path of the image.
        max_size: Maximum size in bytes for each output image.
    
    Returns:
        A base64 string for a static image, or a list of base64 strings for each frame (if animated).
    """

    # Open the image from bytes or file path
    if isinstance(image_data, io.BytesIO):
        pass
    else:
        image_data = io.BytesIO(base64.b64decode(image_data))

    try:
        img = Image.open(image_data)
    
        # Determine number of frames (1 for static images)
        total_frames = getattr(img, "n_frames", 1)
        frames = []
        
        if total_frames > 1:
            # Select up to 5 frame indices (evenly spaced across the animation)
            if total_frames <= split_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = [int(i * (total_frames - 1) / (split_frames - 1)) for i in range(split_frames)]
                frame_indices = sorted(set(frame_indices))  # ensure uniqueness and order
            # Extract the selected frames
            for i in range(total_frames):
                img.seek(i)
                if i in frame_indices:
                    frames.append(img.copy())
                if i == frame_indices[-1]:
                    break  # stop once the last needed frame is extracted
        else:
            frames = [img.copy()]
    except:
        # Try to decode as video
        frames = extract_keyframes(image_data.getvalue(), split_frames)
    
    # Process each frame: convert format, compress, and encode to base64
    output_images = []
    for frame in frames:
        # Decide if we need transparency support
        has_alpha = frame.mode in ("RGBA", "LA") or ('transparency' in frame.info)
        if has_alpha:
            # Use PNG for frames with transparency
            fmt = "PNG"
            processed = frame.convert("RGBA")  # ensure RGBA mode for PNG
            save_params = {"format": "PNG", "optimize": True}
        else:
            # Use JPEG for opaque frames (usually smaller size)
            fmt = "JPEG"
            processed = frame.convert("RGB")  # drop alpha for JPEG
            save_params = {"format": "JPEG", "quality": 85, "optimize": True}
        
        # Save to memory and check size
        buffer = io.BytesIO()
        processed.save(buffer, **save_params)
        data = buffer.getvalue()
        
        # If too large, try to reduce quality (for JPEG) or resize
        if len(data) > max_size:
            if fmt == "JPEG":
                # Reduce JPEG quality in steps
                for q in range(80, 10, -10):  # 80, 70, ..., 20
                    buffer = io.BytesIO()
                    processed.save(buffer, format="JPEG", quality=q, optimize=True)
                    data = buffer.getvalue()
                    if len(data) <= max_size:
                        break  # size is now under the limit
            # If still too large (or if PNG), progressively resize the image
            while len(data) > max_size:
                # Compute new size (reduce to 80% of current dimensions)
                new_w = max(1, int(processed.width * 0.8))
                new_h = max(1, int(processed.height * 0.8))
                if new_w == processed.width or new_h == processed.height:
                    # Cannot resize further (already very small)
                    break
                processed = processed.resize((new_w, new_h), Image.LANCZOS)
                # Save again with current parameters
                buffer = io.BytesIO()
                if fmt == "JPEG":
                    # Ensure mode is RGB after resizing (in case it was altered)
                    processed = processed.convert("RGB")
                    processed.save(buffer, format="JPEG", quality=save_params.get("quality", 85), optimize=True)
                else:
                    processed = processed.convert("RGBA")
                    processed.save(buffer, format="PNG", optimize=True)
                data = buffer.getvalue()
        
        # Encode the final image bytes to base64
        b64_str = base64.b64encode(data).decode('utf-8')
        output_images.append((f"image/{fmt.lower()}", b64_str, len(data)))
    
    # Return a list of strings for multiple frames
    return output_images


__all__ = ['detect_image']