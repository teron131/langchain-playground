import base64
import io
from typing import Tuple, Union

import httpx
from IPython.display import HTML, display
from PIL import Image


def load_image(image_source: str) -> Image.Image:
    """Load an image from a URL or local file."""
    if image_source.startswith(("http://", "https://")):
        with httpx.Client() as client:
            response = client.get(image_source)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
    return Image.open(image_source)


def calculate_new_size(img: Image.Image, max_size: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate the new size of the image based on the max_size."""
    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
    return tuple(int(dim * ratio) for dim in img.size)


def resize_image(img: Image.Image, new_size: Tuple[int, int]) -> Image.Image:
    """Resize the image to the new size."""
    return img.resize(new_size, Image.LANCZOS)


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convert an image to a base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def resize_base64_image(image_source: Union[str, Image.Image], max_size: Tuple[int, int] = (512, 512)) -> str:
    """
    Resize an image from a URL, local file, or PIL Image and return the result as a base64 string.

    Args:
    image_source (str or Image.Image): URL, local file path, or PIL Image of the image to resize.
    max_size (tuple): Desired maximum size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    img = image_source if isinstance(image_source, Image.Image) else load_image(image_source)
    new_size = calculate_new_size(img, max_size)
    resized_img = resize_image(img, new_size)
    return image_to_base64(resized_img, format=img.format or "JPEG")


def plt_img_base64(image_data: str) -> None:
    """Display a base64 encoded image."""
    display(HTML(f'<img src="data:image/jpeg;base64,{image_data}" />'))
