import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 1 if an external webcam is connected
whT = 320  # Width and height for the YOLOv3 model
confidenceThreshold = 0.5
nmsThreshold = 0.3  # Non-Maximum Suppression threshold

# Load class names
classesFile = 'coco.names'
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load YOLO model configuration and weights
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

# Create the network
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:  # Iterate over each detection
            scores = det[5:]  # The first 5 elements are bbox attributes
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Apply Non-Maximum Suppression to avoid overlapping boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, confidenceThreshold, nmsThreshold)

    # Draw bounding boxes for each detected object
    if len(indices) > 0:
        for i in indices.flatten():  # Flatten in case indices is a list of lists
            x, y, w, h = bbox[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the image to blob format
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    findObjects(outputs, img)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
