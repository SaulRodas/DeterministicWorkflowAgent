# src/retrievers/img_retriever.py

import base64
import mimetypes


"""
## IMG Retriever

Al usar un modelo LLM multimodal,
simplemente le pasaremos la imagen.

En caso de no tener uno,
se tendrian que emplear tecnicas de OCR.
"""

def retrieve_img(img_path):

    mime_type, _ = mimetypes.guess_type(img_path)

    if not mime_type:
        mime_type = "image/jpeg"


    with open(img_path, "rb") as f:
        encoded_string = base64.b64encode(
            f.read()
        ).decode("utf-8")


    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded_string}"
        }
    }
