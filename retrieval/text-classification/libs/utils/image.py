from PIL import Image
from torchvision import transforms


def load_image_as_tensor(img_path):
    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    return img
