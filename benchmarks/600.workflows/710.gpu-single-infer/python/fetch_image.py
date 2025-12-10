import os
from uuid import uuid4
from PIL import Image


def handler(event):
    if "image_path" in event:
        return {"image_path": event["image_path"]}

    img = Image.new("RGB", (320, 240), color=(64, 128, 192))
    tmp_path = os.path.join("/tmp", f"{uuid4().hex}.jpg")
    img.save(tmp_path, format="JPEG")
    return {"image_path": tmp_path}
