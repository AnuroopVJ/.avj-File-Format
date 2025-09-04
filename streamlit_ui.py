import streamlit as st
from PIL import Image
import struct
import io

# ---------------------- AVJ Encode/Decode ----------------------
HEADER_FORMAT = '<4s H I I B H B'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

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

def encode_headers(raw_bytes, image_height, image_width, image_mode, alt_text):
    alt_text_encoded = alt_text.encode("utf-8")
    len_alt_text_encoded = len(alt_text_encoded)

    mode_encoded = image_mode.encode("utf-8")
    len_mode_encoded = len(mode_encoded)

    header = struct.pack(
        HEADER_FORMAT,
        b'AVJ1',
        1,
        int(image_height),
        int(image_width),
        3,  # RGB channels
        len_alt_text_encoded,
        len_mode_encoded
    )
    return header + alt_text_encoded + mode_encoded + raw_bytes

def image_to_bytes(image_file):
    img = Image.open(image_file).convert("RGB")
    return img.tobytes(), img.width, img.height, img.mode

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title=".avj Viewer/Encoder", page_icon="📂")
st.title(".avj File Viewer & Encoder")

tab1, tab2 = st.tabs(["Decode .avj", "Encode to .avj"])

# --------- Tab 1: Decode .avj ---------
with tab1:
    uploaded_avj = st.file_uploader("Upload .avj File", type=["avj"])
    if uploaded_avj is not None:
        file_bytes = uploaded_avj.getvalue()
        headers = decode_headers(file_bytes)
        img = reconstruct_image(headers["image_bytes"], headers["width"], headers["height"], headers["mode"])
        
        st.image(img)
        st.write("Metadata:")
        for key, value in headers.items():
            if key != "image_bytes":
                st.write(f"{key}: {value}")

        # allow download as PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download as PNG", data=buf, file_name="decoded.png", mime="image/png")

# --------- Tab 2: Encode to .avj ---------
with tab2:
    uploaded_img = st.file_uploader("Upload Image (.jpg/.png)", type=["jpg", "jpeg", "png"])
    alt_text = st.text_input("Alt Text / Description", value="No description")

    if uploaded_img is not None:
        raw_bytes, w, h, mode = image_to_bytes(uploaded_img)
        encoded_bytes = encode_headers(raw_bytes, h, w, mode, alt_text)

        st.success(f"Image encoded as .avj ({len(encoded_bytes)} bytes)")
        
        st.download_button(
            "Download .avj file",
            data=encoded_bytes,
            file_name="encoded.avj",
            mime="application/octet-stream"
        )
