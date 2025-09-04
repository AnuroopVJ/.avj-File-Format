from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import struct
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel

# ------------------- FastAPI App -------------------
app = FastAPI(title=".avj Encoder/Decoder with Embeddings")

# ------------------- AVJ Format -------------------
HEADER_FORMAT = '<4s H I I B H B'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def image_to_bytes(image_file):
    img = Image.open(image_file).convert("RGB")
    return img.tobytes(), img.width, img.height, img.mode

def encode_headers(raw_bytes, image_height, image_width, image_mode, alt_text):
    alt_text_encoded = alt_text.encode("utf-8")
    len_alt_text_encoded = len(alt_text_encoded)

    mode_encoded = image_mode.encode("utf-8")
    len_mode_encoded = len(mode_encoded)

    header = struct.pack(
        HEADER_FORMAT,
        b'AVJ1',  # magic
        1,        # version
        int(image_height),
        int(image_width),
        3,        # channels RGB
        len_alt_text_encoded,
        len_mode_encoded
    )
    return header + alt_text_encoded + mode_encoded + raw_bytes

def decode_headers(encoded_bytes):
    header = encoded_bytes[:HEADER_SIZE]
    magic, version, height, width, channels, alt_text_len, mode_len = struct.unpack(HEADER_FORMAT, header)

    start = HEADER_SIZE
    alt_text = encoded_bytes[start:start + alt_text_len].decode("utf-8")

    start += alt_text_len
    mode = encoded_bytes[start:start + mode_len].decode("utf-8")

    start += mode_len
    image_bytes = encoded_bytes[start:]

    return {
        "magic": magic.decode("utf-8", errors="ignore"),
        "version": version,
        "height": height,
        "width": width,
        "channels": channels,
        "alt_text": alt_text,
        "mode": mode,
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
    encoded = encode_headers(raw_bytes, h, w, mode, alt_text)

    # Reconstruct PIL for embeddings
    file.file.seek(0)
    pil_img = Image.open(file.file).convert("RGB")
    alt_embedding = embed_alt_text(alt_text)
    img_embedding = embed_image(pil_img)

    buf = io.BytesIO(encoded)
    buf.seek(0)
    return JSONResponse(content={
        "alt_embedding": alt_embedding,
        "image_embedding": img_embedding,
        "avj_file": "Use /download_avj to get the .avj file separately"
    })

@app.post("/download_avj/")
async def download_avj(file: UploadFile = File(...), alt_text: str = "No description"):
    raw_bytes, w, h, mode = image_to_bytes(file.file)
    encoded = encode_headers(raw_bytes, h, w, mode, alt_text)

    buf = io.BytesIO(encoded)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename={file.filename}.avj"
    })

@app.post("/decode/metadata/")
async def decode_metadata(file: UploadFile = File(...)):
    encoded_bytes = await file.read()
    decoded = decode_headers(encoded_bytes)

    # Compute embeddings
    img = reconstruct_image(decoded["image_bytes"], decoded["width"], decoded["height"], decoded["mode"])
    decoded.pop("image_bytes")
    decoded["alt_embedding"] = embed_alt_text(decoded["alt_text"])
    decoded["image_embedding"] = embed_image(img)

    return JSONResponse(content=decoded)

@app.post("/decode/image/")
async def decode_image(file: UploadFile = File(...)):
    encoded_bytes = await file.read()
    decoded = decode_headers(encoded_bytes)
    img = reconstruct_image(decoded["image_bytes"], decoded["width"], decoded["height"], decoded["mode"])

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png", headers={
        "Content-Disposition": "attachment; filename=decoded.png"
    })