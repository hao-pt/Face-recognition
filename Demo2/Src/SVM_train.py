from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, required=True,
	help='path to serialized db of facial encodings')
args = vars(ap.parse_args())

# Load data
data = pickle.loads(open(args['dataset'], "rb").read())

# Prepare X and y
X,y = data["encodings"],data["names"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train ...
clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X,y)

# Write to file
f = open('trainedSVM.pickle', "wb")
f.write(pickle.dumps(clf))
f.close()

f = open('label.pickle', "wb")
f.write(pickle.dumps(label_encoder))
f.close()


#print(data['encodings'])
print(data["names"])