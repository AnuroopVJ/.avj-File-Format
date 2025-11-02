# `.avj Encoder/Decoder with CLIP Embeddings`

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.101-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-orange)

A Python application to **encode images into a custom `.avj` format with auto-generated CLIP embeddings**. The project includes:

* **FastAPI backend** for programmatic access
* **Streamlit frontend** for interactive encoding/decoding
* **Lazy-loading CLIP models** for performance
* **Zstandard compression** for image bytes

---

## Features

* **Custom `.avj` format**: Stores image bytes, metadata, and CLIP embeddings in a single file.
* **Encode / Decode Images**: Convert images to `.avj` and back to standard formats (PNG).
* **CLIP Embeddings**: Generates embeddings for alt text and image for AI/ML applications.
* **Compression**: Zstandard compression for smaller `.avj` file sizes.
* **Lazy Loading**: CLIP models load only when needed, reducing startup time.
* **Streamlit Viewer**: Interactive UI to upload, view, encode, and decode `.avj` files.
* **FastAPI Endpoints**: Programmatic access to all features via REST API.

---

## Project Structure

```
AI_agentic_software_dev/
│
├─ api/                  # FastAPI endpoints
│   └─ main.py
│
├─ encode_decode/        # Core logic for encoding, decoding, embeddings, and compression
│   └─ encode_decode.py
│
├─ streamlit_ui/         # Streamlit frontend
│   └─ main.py
│
├─ requirements.txt      # Python dependencies
└─ README.md             # This file
```

---

## `.avj` File Format

The `.avj` file contains:

1. **Header** (fixed-size + dynamic lengths):

| Field                       | Description                        |
| --------------------------- | ---------------------------------- |
| Magic (`AVJ1`)              | File identifier                    |
| Version (`1`)               | Format version                     |
| Height, Width               | Image dimensions                   |
| Channels                    | Number of channels (RGB = 3)       |
| Alt Text Length             | Bytes of UTF-8 alt text            |
| Mode Length                 | Bytes of image mode (`RGB`)        |
| Alt Embedding Length        | Float32 size of alt-text embedding |
| Image Embedding Length      | Float32 size of image embedding    |
| Compression Flag (optional) | 1 if compressed, 0 if raw          |

2. **Payload**:

* Alt Text (UTF-8)
* Image Mode (UTF-8)
* Alt Text Embedding (float32)
* Image Embedding (float32)
* Raw or Compressed Image Bytes

---

## Installation

```bash
git clone <your_repo_url>
cd AI_agentic_software_dev
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Running the App

### 1. FastAPI Backend

```bash
uvicorn api.main:app --reload
```

**Endpoints**:

| Endpoint            | Method | Description                                        |
| ------------------- | ------ | -------------------------------------------------- |
| `/encode/`          | POST   | Upload image + alt text → Returns `.avj` file      |
| `/decode/metadata/` | POST   | Upload `.avj` → Returns metadata + embeddings      |
| `/decode/image/`    | POST   | Upload `.avj` → Returns reconstructed PNG          |
| `/compress/`        | POST   | Upload `.avj` → Returns Zstandard-compressed bytes |


### 2. Streamlit Frontend

```bash
streamlit run streamlit_ui/main.py
```

**Features**:

* **Encode to `.avj`**: Upload an image, enter alt text, download `.avj`.
* **Decode `.avj`**: Upload `.avj`, view metadata, download PNG.
* Compressed images use **Zstandard** for faster storage and smaller file sizes.
---

## 🔹 Example Workflow (Python)

### Encode an image

```python
import requests

with open("example.png", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/encode/",
        files={"file": f},
        data={"alt_text": "Example image"}
    )

with open("example.avj", "wb") as out:
    out.write(response.content)
```

### Decode metadata

```python
with open("example.avj", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/decode/metadata/",
        files={"file": f}
    )
print(response.json())
```

### Decode image

```python
with open("example.avj", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/decode/image/",
        files={"file": f}
    )
with open("decoded.png", "wb") as out:
    out.write(response.content)
```

---



---

## Needed Improvements

* Compress embeddings for further `.avj` size reduction.
* Batch encoding/decoding for multiple images.
* CLIP-based search over `.avj` files.
* Add encryption for secure storage.

---

## License

MIT License © 2025

---

## 🙏 Acknowledgements

* [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
* Streamlit & FastAPI communities for web frameworks

