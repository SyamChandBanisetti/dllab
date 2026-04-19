from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
model = YOLO("yolov8n.pt") # auto downloads
image_path = "play.jpg" # your uploaded image
results = model(image_path)
output_img = results[0].plot()
plt.imshow(output_img)
plt.axis("off")
plt.title("Detected Objects")
plt.show()
cv2.imwrite("output.jpg", output_img)