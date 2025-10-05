import os
import base64
import io
from typing import Optional
from PIL import Image

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Optional: init_chat_model provides a cross-provider "invoke" that accepts
# content-block style multimodal messages (text + image/file/audio blocks).
try:
    from langchain.chat_models import init_chat_model  # type: ignore
except Exception:
    init_chat_model = None


def _encode_image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_llm_response(
    prompt: str,
    model: str = "gpt-4.1",
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> str:
    """Gets a response from an LLM using LangChain.

    - Uses `ChatOpenAI` (OpenAI API) when an OpenAI-style model is requested.
    - If a Gemini model name is supplied and langchain provides a Gemini client,
      it will attempt to use it; otherwise an informative error is returned.

    The image, if provided, is encoded as a data URI and appended to the
    user message so the model can reference it. Depending on your model/provider,
    you may want a different image handling strategy (upload + URL).
    """
    try:
        # Configure API keys in env for provider SDKs used by LangChain
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key

        system_text = "You are an expert assistant that extracts and summarizes contact information. Respond concisely."

        user_text = prompt or ""
        if image is not None:
            if not isinstance(image, Image.Image):
                return "Provided image is not a PIL Image instance."
            try:
                data_uri = _encode_image_to_data_uri(image)
                user_text += f"\n\n[Attached image as data URI] {data_uri}"
            except Exception as ie:
                return f"Error encoding image: {ie}"

        # Prefer init_chat_model if available; it supports the cross-provider
        # multimodal content-block format described in the docs.
        if init_chat_model is not None:
            provider_model = model
            # normalize model string into provider:id format if necessary
            if model and model.lower().startswith("gpt") and not model.lower().startswith("openai:"):
                provider_model = f"openai:{model}"
            if model and "gemini" in model.lower() and not model.lower().startswith("google_genai:"):
                provider_model = f"google_genai:{model}"

            try:
                llm = init_chat_model(provider_model)
                # build the message in content-block multimodal format
                content_blocks = []
                content_blocks.append({"type": "text", "text": user_text})
                if image is not None:
                    try:
                        if not isinstance(image, Image.Image):
                            return "Provided image is not a PIL Image instance."
                        buf = io.BytesIO()
                        image.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        content_blocks.append({
                            "type": "image",
                            "source_type": "base64",
                            "mime_type": "image/png",
                            "data": b64,
                        })
                    except Exception as ie:
                        return f"Error encoding image: {ie}"

                message = {"role": "user", "content": content_blocks}
                resp = llm.invoke([message])
                # Some providers expose .text() on the response, others return
                # a richer object. Try common access patterns then fallback.
                try:
                    if hasattr(resp, "text"):
                        return resp.text()
                    if isinstance(resp, str):
                        return resp
                    # If it's a sequence-like with last element containing text
                    if hasattr(resp, "content"):
                        return getattr(resp, "content")
                    # Fallback to string representation
                    return str(resp)
                except Exception:
                    return str(resp)
            except Exception as e:
                return f"init_chat_model invocation failed: {e}"

        # Fallback: use ChatOpenAI with plain-text prompt (image will be embedded as data URI)
        try:
            if model and model.lower().startswith("gpt"):
                chat = ChatOpenAI(model=model, temperature=0.1)
            else:
                chat = ChatOpenAI(temperature=0.1)

            resp = chat([SystemMessage(content=system_text), HumanMessage(content=user_text)])
            # LangChain chat models usually return an AIMessage-like object
            if hasattr(resp, "content"):
                return getattr(resp, "content")
            return str(resp)
        except Exception as e:
            return f"ChatOpenAI call failed: {e}"
    except Exception as e:
        return f"An error occurred: {e}"
