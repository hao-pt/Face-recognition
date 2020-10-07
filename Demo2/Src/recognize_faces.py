# import some necessary packages
import tensorflow as tf
import face_recognition
import argparse
import pickle
import cv2
 
from ServiceMTCNN import detect_face as lib

sess = tf.Session()
pnet, rnet, onet = lib.create_mtcnn(sess, None)

def extract_faces_from_image(input_img,pnet,rnet,onet): 
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    print("[+] Initialized MTCNN modules")

    boxes, landmarks = lib.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor)

    faces = []
    bbs = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(
            box[1]), int(box[2]), int(box[3])
        if y1 < 0:
            y1 = 0
        if x1 < 0:
            x1 = 0
        if y2 > img.shape[0]:
            y2 = img.shape[0]
        if x2 > img.shape[1]:
            x2 = img.shape[1]
        faces.append(input_img[y1:y2, x1:x2])
        #diff = (y2-y1) - (x2-x1)
        #bbs.append((x1, int(y1 + diff/2), x2, int(y2 - diff/2))) # left,top,right,bottom
        #bbs.append((int(y1 + diff/2), x2, int(y2 - diff/2), x1)) # top,right,bottom,left
        bbs.append((y1,x2,y2,x1))
    return faces, bbs

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--encodings', type=str, required=True,
	help='path to serialized db of facial encodings')
ap.add_argument('--image', type=str, required=True,
	help='path to input image')
ap.add_argument('--detection-method', type=str, default='cnn',
	help='face detection model to use: either `hog` or `cnn`')
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args['encodings'], "rb").read())
 
# load the input image and convert it from BGR to RGB
image = cv2.imread(args['image'])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
#boxes = face_recognition.face_locations(rgb, model=args['detection_method'])
boxes = extract_faces_from_image(image,pnet,rnet,onet)[1]
print(boxes)
encodings = face_recognition.face_encodings(rgb, boxes)
 
# initialize the list of names for each face detected
names = []

# Load trained SVM model and label encoder
trainedSVM = pickle.loads(open("trainedSVM.pickle", "rb").read())
label_encoder = pickle.loads(open("label.pickle", "rb").read())

for encoding in encodings:
    encoding = encoding.reshape(1,-1)
    name = "Unknown"
    predicted_class = trainedSVM.predict(encoding)
    print(predicted_class)
    name = label_encoder.classes_[predicted_class]
    print(name)
    names.append(name[0])


# loop over the recognized faces
for ((top,right,bottom,left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (0, 255, 0), 2)
 
# show the output image
cv2.imwrite('output_'+args['image'][-6:-4]+'.jpg',image)
cv2.imshow('Image', image)
cv2.waitKey(0)

