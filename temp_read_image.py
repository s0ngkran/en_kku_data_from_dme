import cv2
from PIL import Image
from torchvision import transforms

img_path = './dog.jpg'

img_cv2 = cv2.imread(img_path)
img_io = Image.open(img_path)
print('cv2', img_cv2)
print()
print()
print()
print()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
image_tensor = preprocess(img_io)
print('io', image_tensor)
print('')
print('')
print('')
print('')
print('')
