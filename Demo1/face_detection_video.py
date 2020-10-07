import numpy as np
import cv2


def face_detection(img, configFile, modelFile, showResult = False):
    # Load pretrained Caffe model
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    # Get size of image
    (height, weight) = img.shape[:2]

    # Prepare imput image as 4-D blob to feed to network
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), # Source image
            1.0, # Scale factor
            (300, 300), # Size default
            (104.0, 177.0, 123.0), # Specify mean that will be subtracted from each image channel to reduce ilummination changes
            swapRB = True, crop = False) # Convert BGR to RGB for processing

    # Forward pass through the network
    net.setInput(blob)
    detections = net.forward()

    # Minimal confidence to threshold
    minimum_conf = 0.5

    # Store scores and bounding box of each object
    conf_list = []
    box_list = []

    # Loop over detections to draw bounding box around face region
    for i in range(0, detections.shape[2]):
        # Get confidence (probability, score) of predicted face
        confidence = detections[0, 0, i, 2]
        

        # Filter out week detection of detected face base on confidence (> 0.5)
        if confidence > minimum_conf:
            # Get (x, y) coordinate of bounding box for detected face
            # Because the output coordinates of bounding box are normalized between [0, 1]
            # Thus we should multiply by height and width of original image
            # To get correct coordinates
            bounding_box = detections[0, 0, i, 3:7] * np.array([weight, height, weight, height])
            (startX, startY, endX, endY) = bounding_box.astype("int")
            box_list.append((startX, startY, endX, endY)) # store
            conf_list.append(confidence) # store

            # Draw bounding box with probability
            text = "{:.2f}%".format(confidence * 100)
            # Check if y coordinate has enough 10 space to draw text
            y = startY - 10 if startY - 10 > 10 else startY + 10
            # Draw box
            cv2.rectangle(img, (startX, startY), (endX, endY), (0,0,255), 2)
            # Put text (Score)
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    if showResult:
        # Display image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Face detection"); plt.axis("off")
        plt.show()

    return (conf_list, box_list) # Return score and bounding box of all object in image

# get reference of webcam (or video)
video_capture = cv2.VideoCapture(0)

# Define config and model file for detecting
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"

process_this_frame = True

while True:
    # Get a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster processing
    resized_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
    
    # Just detect when flag is on to save time
    if process_this_frame:
        # Get scores and boxes
        (scores, boxes) = face_detection(resized_frame, configFile, modelFile, showResult = False)
        
    # Negate this flag
    process_this_frame = not process_this_frame

    # Draw confidence and bounding box
    for i in range(len(scores)):
        print(i)
        # Get confidence
        confidence = scores[i]
        # Get box coordinates
        startX, startY, endX, endY = np.array(boxes[i]) * 4 # Scale back 4 to get correct coordinates

        # Draw bounding box with probability
        text = "{:.2f}%".format(confidence * 100)
        # Check if y coordinate has enough 10 space to draw text
        y = startY - 10 if startY - 10 > 10 else startY + 10
        # Draw box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
        # Put text (Score)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display
    cv2.imshow("Face detection", frame)
    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
    
# Release handle on webcam
video_capture.release()
cv2.destroyAllWindows()

