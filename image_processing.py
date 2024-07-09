import base64
import io

import httpx
from IPython.display import HTML, display
from PIL import Image


def resize_base64_image(image_source, max_size=(512, 512)):
    """
    Resize an image from a URL or local file and return the result as a base64 string.

    Args:
    image_source (str): URL or local file path of the image to resize.
    max_size (tuple): Desired maximum size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    if image_source.startswith(("http://", "https://")):
        with httpx.Client() as client:
            response = client.get(image_source)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
    else:
        img = Image.open(image_source)

    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
    new_size = tuple(int(dim * ratio) for dim in img.size)

    resized_img = img.resize(new_size, Image.LANCZOS)

    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format or "JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def plt_img_base64(image_data):
    display(HTML(f'<img src="data:image/jpeg;base64,{image_data}" />'))
