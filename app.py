

import streamlit as st
from llm import get_llm_response
from streamlit_paste_button import paste_image_button as pbutton
from PIL import Image
import io
import base64
import io
import requests
from PIL import Image

def main():
    st.title("LLM Interaction App")

    # Sidebar for API keys
    with st.sidebar:
        st.header("API Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        gemini_api_key = st.text_input("Google/Gemini API Key", type="password")
        model_choice = st.selectbox(
            "Choose a model",
            [
                "gpt-3.5-turbo",
                "gpt-4o",
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-5",
                "gpt-5-mini",
                "gemini/gemini-pro"
            ]
        )

    # Input field
    prompt = st.text_input("Enter your prompt here:")

    st.markdown("**Attach an image to send to the LLM:**")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        paste_result = pbutton("📋 Paste an image")
        pasted_image = paste_result.image_data if paste_result and paste_result.image_data is not None else None
    with col_img2:
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"])
        uploaded_image = None
        if uploaded_file:
            try:
                uploaded_image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error loading uploaded image: {e}")

    # Choose which image to send (prefer pasted)
    image_to_send = pasted_image if pasted_image is not None else uploaded_image
    if image_to_send is not None:
        st.image(image_to_send, caption="Image to send to LLM")

    # Submit button
    if st.button("Submit"):
        if not prompt and image_to_send is None:
            st.warning("Please enter a prompt or attach an image.")
        else:
            with st.spinner("Getting response from LLM..."):
                response = get_llm_response(
                    prompt=prompt,
                    model=model_choice,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    image=image_to_send
                )
                st.success("Response received!")
                st.write(response)

    st.markdown("---")



def _load_image_from_paste(pasted: str):
    try:
        pasted = pasted.strip()
        if not pasted:
            return None
        if pasted.startswith("data:image"):
            header, b64 = pasted.split(",", 1)
            img_bytes = base64.b64decode(b64)
            return Image.open(io.BytesIO(img_bytes))
        if pasted.startswith("http://") or pasted.startswith("https://"):
            resp = requests.get(pasted, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        # raw base64
        img_bytes = base64.b64decode(pasted, validate=True)
        return Image.open(io.BytesIO(img_bytes))
    except Exception:
        return None



if __name__ == "__main__":
    main()
        