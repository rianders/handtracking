import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from cvzone.PoseModule import PoseDetector
import socket

# Load the DeepLabv3 model for human segmentation
model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
model.eval()

# Define the transform for the input image
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def segment_human(image, model, resize_shape=(224, 224)):
    # Resize the input image for segmentation
    image_resized = cv2.resize(image, resize_shape)
    # Convert OpenCV image (NumPy array) to PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    
    input_image = trf(image_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    # Convert the mask to uint8 data type and resize it back to the original image size
    mask = (output_predictions == 15).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
success, img = cap.read()
h, w, _ = img.shape
detector = PoseDetector()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    mask = segment_human(img, model)

    # Apply the mask to the original image
    masked_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))

    img = detector.findPose(masked_img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)

    data = []

    if bboxInfo:
        center = bboxInfo["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        for lm in lmList:
            data.extend([lm[0], h - lm[1], lm[2]])

        sock.sendto(str.encode(str(data)), serverAddressPort)

    # Display the original image without resizing
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
