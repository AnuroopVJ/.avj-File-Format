from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import struct
from PIL import Image
import io
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# ------------------- FastAPI App -------------------
app = FastAPI(title=".avj Encoder/Decoder with Embeddings")

# ------------------- AVJ Format -------------------
# Header now includes lengths of alt_text embedding and image embedding
HEADER_FORMAT = '<4s H I I B H B I I'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def image_to_bytes(image_file):
    img = Image.open(image_file).convert("RGB")
    return img.tobytes(), img.width, img.height, img.mode

def encode_headers_with_embeddings(raw_bytes, h, w, mode, alt_text, alt_emb, img_emb):
    alt_text_encoded = alt_text.encode("utf-8")
    len_alt_text_encoded = len(alt_text_encoded)

    mode_encoded = mode.encode("utf-8")
    len_mode_encoded = len(mode_encoded)

    alt_emb_bytes = np.array(alt_emb, dtype=np.float32).tobytes()
    img_emb_bytes = np.array(img_emb, dtype=np.float32).tobytes()

    header = struct.pack(
        HEADER_FORMAT,
        b'AVJ1',        # magic
        1,              # version
        int(h),
        int(w),
        3,              # channels RGB
        len_alt_text_encoded,
        len_mode_encoded,
        len(alt_emb_bytes),
        len(img_emb_bytes)
    )

    return header + alt_text_encoded + mode_encoded + alt_emb_bytes + img_emb_bytes + raw_bytes

def decode_headers_with_embeddings(encoded_bytes):
    header = encoded_bytes[:HEADER_SIZE]
    magic, version, height, width, channels, alt_text_len, mode_len, alt_emb_len, img_emb_len = struct.unpack(HEADER_FORMAT, header)

    start = HEADER_SIZE
    alt_text = encoded_bytes[start:start+alt_text_len].decode("utf-8")
    start += alt_text_len

    mode = encoded_bytes[start:start+mode_len].decode("utf-8")
    start += mode_len

    alt_emb_bytes = encoded_bytes[start:start+alt_emb_len]
    alt_embedding = np.frombuffer(alt_emb_bytes, dtype=np.float32)
    start += alt_emb_len

    img_emb_bytes = encoded_bytes[start:start+img_emb_len]
    image_embedding = np.frombuffer(img_emb_bytes, dtype=np.float32)
    start += img_emb_len

    image_bytes = encoded_bytes[start:]

    return {
        "magic": magic.decode("utf-8", errors="ignore"),
        "version": version,
        "height": height,
        "width": width,
        "channels": channels,
        "alt_text": alt_text,
        "mode": mode,
        "alt_embedding": alt_embedding.tolist(),
        "image_embedding": image_embedding.tolist(),
        "image_bytes": image_bytes
    }

def reconstruct_image(image_bytes, width, height, mode="RGB"):
    return Image.frombytes(mode, (width, height), image_bytes)

# ------------------- CLIP Embeddings -------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_alt_text(text: str):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features[0].cpu().numpy().tolist()

def embed_image(pil_image: Image.Image):
    inputs = clip_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features[0].cpu().numpy().tolist()

# ------------------- FastAPI Endpoints -------------------

@app.post("/encode/")
async def encode_image(file: UploadFile = File(...), alt_text: str = "No description"):
    raw_bytes, w, h, mode = image_to_bytes(file.file)
    file.file.seek(0)
    pil_img = Image.open(file.file).convert("RGB")

    alt_emb = embed_alt_text(alt_text)
    img_emb = embed_image(pil_img)

    encoded = encode_headers_with_embeddings(raw_bytes, h, w, mode, alt_text, alt_emb, img_emb)

    buf = io.BytesIO(encoded)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename={file.filename}.avj"
    })


@app.post("/decode/metadata/")
async def decode_metadata(file: UploadFile = File(...)):
    encoded_bytes = await file.read()
    decoded = decode_headers_with_embeddings(encoded_bytes)

    # Exclude raw image bytes from metadata response
    decoded_meta = decoded.copy()
    decoded_meta.pop("image_bytes")
    return JSONResponse(content=decoded_meta)


@app.post("/decode/image/")
async def decode_image(file: UploadFile = File(...)):
    encoded_bytes = await file.read()
    decoded = decode_headers_with_embeddings(encoded_bytes)
    img = reconstruct_image(decoded["image_bytes"], decoded["width"], decoded["height"], decoded["mode"])

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png", headers={
        "Content-Disposition": "attachment; filename=decoded.png"
    })
