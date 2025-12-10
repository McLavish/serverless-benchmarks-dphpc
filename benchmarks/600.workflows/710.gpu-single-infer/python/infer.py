import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

_model = torchvision.models.resnet18(weights="DEFAULT").cuda().eval()
_transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def handler(event):
    path = event["image_path"]
    img = Image.open(path).convert("RGB")
    x = _transform(img).unsqueeze(0).cuda()

    with torch.inference_mode():
        logits = _model(x)
    top_idx = int(torch.argmax(logits, dim=1).item())
    score = float(torch.nn.functional.softmax(logits, dim=1)[0, top_idx].item())

    return {"top1_idx": top_idx, "top1_score": score, "image_path": path}
