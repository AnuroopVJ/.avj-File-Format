import streamlit as st
from PIL import Image
import io
from main import (
    encode_headers_with_embeddings,
    decode_headers_with_embeddings,
    image_to_bytes,
    reconstruct_image,
    embed_alt_text,
    embed_image
)

st.set_page_config(page_title=".avj Viewer/Encoder", page_icon="📂")
st.title(".avj File Viewer & Encoder with Embeddings")

tab1, tab2 = st.tabs(["Decode .avj", "Encode to .avj"])

# --------- Tab 1: Decode .avj ---------
with tab1:
    uploaded_avj = st.file_uploader("Upload .avj File", type=["avj"])
    if uploaded_avj:
        file_bytes = uploaded_avj.getvalue()
        headers = decode_headers_with_embeddings(file_bytes)
        img = reconstruct_image(headers["image_bytes"], headers["width"], headers["height"], headers["mode"])

        st.image(img)
        st.write("Metadata and Embeddings:")
        for k, v in headers.items():
            if k != "image_bytes":
                st.write(f"{k}: {v}")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download as PNG", data=buf, file_name="decoded.png", mime="image/png")

# --------- Tab 2: Encode to .avj ---------
with tab2:
    uploaded_img = st.file_uploader("Upload Image (.jpg/.png)", type=["jpg", "jpeg", "png"])
    alt_text = st.text_input("Alt Text / Description", value="No description")

    if uploaded_img:
        raw_bytes, w, h, mode = image_to_bytes(uploaded_img)
        pil_img = Image.open(uploaded_img).convert("RGB")
        alt_emb = embed_alt_text(alt_text)
        img_emb = embed_image(pil_img)

        encoded_bytes = encode_headers_with_embeddings(raw_bytes, h, w, mode, alt_text, alt_emb, img_emb)
        st.success(f"Image encoded as .avj ({len(encoded_bytes)} bytes)")

        st.download_button(
            "Download .avj file",
            data=encoded_bytes,
            file_name="encoded.avj",
            mime="application/octet-stream"
        )
