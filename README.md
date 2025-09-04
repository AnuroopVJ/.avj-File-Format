# .avj Encoder/Decoder with CLIP Embeddings

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.101-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-orange)

A Python application to encode images into a custom `.avj` format, decode them, and generate embeddings using OpenAI's CLIP model. It includes both a **Streamlit frontend** for interactive use and a **FastAPI backend** for programmatic access.



## Features

* **Custom `.avj` format**: Stores image bytes with metadata including width, height, channels, mode, and alt text.
* **Encode / Decode Images**: Convert images to `.avj` and back to standard image formats (PNG).
* **CLIP Embeddings**: Generate embeddings for both the image and its alt text for AI/ML applications.
* **Streamlit Viewer**: Interactive UI to upload, view, and download `.avj` files or images.
* **FastAPI API**: Endpoints for encoding, decoding, and retrieving embeddings programmatically.

---

## `.avj` Format

The `.avj` file consists of:

* **Header** (fixed size, 16 bytes + dynamic lengths)

  * Magic: `AVJ1`
  * Version: `1`
  * Image Height, Width
  * Channels (RGB)
  * Alt Text Length
  * Mode Length
* **Alt Text** (UTF-8)
* **Mode** (UTF-8)
* **Raw Image Bytes** (RGB)

---

## Requirements

```bash
Python 3.12+
pip install streamlit fastapi uvicorn pillow torch transformers
```

---

## Streamlit Usage

1. Run the Streamlit app:

```bash
streamlit run streamlit_ui.py
```

2. **Tabs**:

   * **Decode `.avj`**: Upload a `.avj` file, view the image and metadata, download as PNG.
   * **Encode to `.avj`**: Upload an image, provide alt text, and download the `.avj` file.

---

## FastAPI Usage

1. Start the FastAPI server:

```bash
uvicorn streamlit_ui:app --reload
```

2. **Endpoints**:

| Endpoint            | Method | Description                                                                   |
| ------------------- | ------ | ----------------------------------------------------------------------------- |
| `/encode/`          | POST   | Upload image and alt text → Returns CLIP embeddings and `.avj` data reference |
| `/download_avj/`    | POST   | Download `.avj` file for a given image and alt text                           |
| `/decode/metadata/` | POST   | Upload `.avj` → Returns metadata + CLIP embeddings                            |
| `/decode/image/`    | POST   | Upload `.avj` → Returns the decoded image as PNG                              |

**Example with `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/encode/" \
-F "file=@example.png" \
-F "alt_text=An example image"
```

---

## Example Workflow

1. Encode an image to `.avj` with alt text:

```python
from PIL import Image
import requests

with open("example.png", "rb") as f:
    response = requests.post("http://127.0.0.1:8000/encode/", files={"file": f}, data={"alt_text": "Example image"})
print(response.json())
```

2. Decode `.avj` and retrieve image/metadata:

```python
with open("example.avj", "rb") as f:
    response = requests.post("http://127.0.0.1:8000/decode/metadata/", files={"file": f})
print(response.json())
```

---

## Folder Structure

```
├── streamlit_ui.py    # Streamlit UI
├── main.py          # FastAPI backend
├── README.md               # Documentation
└── requirements.txt        # Dependencies
```

---

## License

MIT License © 2025

---

## Acknowledgements

* OpenAI [CLIP Model](https://huggingface.co/openai/clip-vit-base-patch32)
* Streamlit & FastAPI communities for web frameworks

---
