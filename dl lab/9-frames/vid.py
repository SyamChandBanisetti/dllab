import cv2

# Load video
video = cv2.VideoCapture('input.mp4')

count = 0

while True:
    success, frame = video.read()

    if not success:
        break

    # Save each frame as image
    cv2.imwrite(f"frame_{count}.jpg", frame)

    count += 1

video.release()
print("Frames extracted successfully!")