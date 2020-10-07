from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import tensorflow as tf

from ServiceMTCNN import detect_face as lib


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, required=True,
	help='path to input directory of faces + images')
ap.add_argument('--encodings', type=str, required=True,
	help='path to serialized db of facial encodings')
ap.add_argument('--detection_method', type=str, default='cnn',
	help='face detection model to use: either `hog` or `cnn`')
args = vars(ap.parse_args())

sess = tf.Session()
pnet, rnet, onet = lib.create_mtcnn(sess, None)
def extract_faces_from_image(input_img, pnet, rnet, onet):
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
        diff = (y2-y1) - (x2-x1)
        #bbs.append((x1, int(y1 + diff/2), x2, int(y2 - diff/2)))
        bbs.append((int(y1 + diff/2), x2, int(y2 - diff/2), x1)) # top,right,bottom,left
		#bbs.append((y1,x1))
    return faces, bbs

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args['dataset']))
 
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print(imagePath)
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	print(name)
	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	#boxes = face_recognition.face_locations(rgb, model=args['detection_method'])
	boxes = extract_faces_from_image(image, pnet, rnet,onet)[1]
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
 
	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()