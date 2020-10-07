# Import libraries
import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils

from imutils.face_utils import FaceAligner # Face alignment
from imutils.face_utils import rect_to_bb

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

    # Copy image
    copy_img = np.copy(img)

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
            cv2.rectangle(copy_img, (startX, startY), (endX, endY), (0,0,255), 2)
            # Put text (Score)
            cv2.putText(copy_img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    if showResult:
        cv2.imshow("Detected face", copy_img)

    return (conf_list, box_list) # Return score and bounding box of all object in image
    
def facial_landmark_and_alignment(img, boxes, isShow = False):
    # Create facial landmark predictor
    # predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Init face aligner
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # Grayscale image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Face aligned
    # faces = []

    faceAligned = []

    # Loop over bounding box
    for box in boxes:
        rect = dlib.rectangle(left=box[0], top=box[1], right=box[2], bottom=box[3])
        # Determine facial landmarks for face region
        landmarks = predictor(grayImg, rect)
        
        # Convert facial landmark (x, y) into numpy array
        landmarks = face_utils.shape_to_np(landmarks)

        # Loop over facial landmarks and draw them
        for (i, (x, y)) in enumerate(landmarks):
            if isShow:
                cv2.circle(img, (x, y), 1, (0,0,255), -1) # -1: fill circle will be drawn
                # Draw text
                cv2.putText(img, str(i+1), (x - 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # extract the ROI of face
            (x, y, w, h) = rect_to_bb(rect)

            # Resize region with width = 256
            faceOrig = imutils.resize(img[y:y+h, x:x+w], width = 256)
            # Align face image
            faceAligned = fa.align(img, grayImg, rect)
            
            # # Store this face
            # faces.append(faceAligned)

            if isShow:
                cv2.imshow("Original face", faceOrig)
                cv2.imshow("Aligned face", faceAligned)

        if isShow:
            # Display image
            cv2.imshow("Facial Landmark", img)

        return faceAligned

def main():
    # Define config and model file for detecting
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"

    # Read image
    img = cv2.imread(r"E:\K16\Junior\Face-recognition\1612174\dataset\justin_bieber\00008.jpg")
    H, W = img.shape[:2]

    # Resize image: width = 800
    img = imutils.resize(img, width=800)

    # -------------------------------------Dlib-------------------------------------------
    # # Convert to gray image
    # grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Detect with dlib
    # detector = dlib.get_frontal_face_detector()
    # rects = detector(grayImg, 2)

    # # Create facial landmark predictor
    # # predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # # Init face aligner
    # fa = FaceAligner(predictor, desiredFaceWidth=256)

    # # loop over the face detections
    # for rect in rects:
    #     # extract the ROI of the *original* face, then align the face
    #     # using facial landmarks
    #     (x, y, w, h) = rect_to_bb(rect)
    #     faceOrig = imutils.resize(img[y:y + h, x:x + w], width=256)
    #     faceAligned = fa.align(img, grayImg, rect)
    
    #     # display the output images
    #     cv2.imshow("Original", faceOrig)
    #     cv2.imshow("Aligned", faceAligned)
    #     cv2.waitKey(0)

    # ---------------------------- OpenCV detector ----------------------------------------
    # Localize face in image
    scores, boxes = face_detection(img, configFile, modelFile, False)

    # Make sure exist detected face
    if len(scores) > 0:
        # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = facial_landmark_and_alignment(img, boxes, True)

        # cv2.imshow("sdsd", face)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# main()