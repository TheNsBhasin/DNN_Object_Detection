# Import necessary libraries
import numpy as np
import argparse
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network')
ap.add_argument('-v', '--video', type=str, default='',
                help='Path to video file. If empty, web cam stream will be used')
ap.add_argument('-p', '--prototxt', required=True,
                help="Path to Caffe 'deploy' prototxt file")
ap.add_argument('-m', '--model', required=True,
                help='Path to weights for Caffe model')
ap.add_argument('-l', '--labels', required=True,
                help='Path to labels for dataset')
ap.add_argument('-c', '--confidence', type=float, default=0.2,
                help='Minimum probability to filter weak detections')
args = vars(ap.parse_args())

# Initialize class labels of the dataset
CLASSES = [line.strip() for line in open(args['labels'])]
print('[INFO]', CLASSES)

# Generate random bounding box colors for each class label
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load Caffe model from disk
print("[INFO] Loading model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Open video capture from file or capture device
print("[INFO] Starting video stream")
if args['video']:
    cap = cv2.VideoCapture(args['video'])
else:
    cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    (h, w) = frame.shape[:2]

    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size.
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # Extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Draw bounding box for the object
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_id], 2)

            # Draw label and confidence of prediction in frame
            label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[class_id], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 2)

    # Show fame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

# Clean-up
cap.release()
cv2.destroyAllWindows()
