import cv2
from torchvision import datasets, transforms
from PIL import Image

def MyLoader(image_path):


    image = cv2.imread(image_path)  # 使用OpenCV的imread函数加载图像
    image = Image.fromarray(image)
    return image




transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])