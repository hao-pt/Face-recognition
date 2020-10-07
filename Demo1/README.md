# Folder structure:
`dataset`: training dataset which folder name is name/identity of a object.

`face_detection_model`: stores model configuration and pretrained weights of SSD.

`ISR`: image super resolution module. We didnt integrate it into out source because ISR took long time to produre results. To play with it: 
- Firstly, you need to clone ISR: `git clone https://github.com/idealo/image-super-resolution` and use [image_super_resolution.py](ISR/image_super_resolution.py) script to run.
- [Pretrained model](ISR/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5) are also provided.


`output`: contains some pickle files
- [embeddings.pickle](output/embeddings.pickle): stores embedding images or 128-d feature vectors of training dataset.
- [le.pickle](output/le.pickle): corresponding labels for embedding images.
- [recognizer.pickle](output/recognizer.pickle): stores trained model of SVM for face recognition.

# Installation
```
pip install -r requirements.txt
```

# Usage:
Face recognition for static image:
```
python recognize.py -i test_images/obama_1_hi.jpg \ # path to image 
        -d ./face_detection_model/ \ # path to OpenCV's face detector  
        -m ./openface_nn4.small2.v1.t7 \ # path to OpenCV's face embedding model 
        -r ./output/recognizer.pickle \ # SVM model for face recognition 
        -l ./output/le.pickle \ # labels
        -c 0.5 # face threshold
```
Face recognition for video:
```
python recognize.py -i path/to/video 
        -d ./face_detection_model/ \ # path to OpenCV's face detector  
        -m ./openface_nn4.small2.v1.t7 \ # path to OpenCV's face embedding model 
        -r ./output/recognizer.pickle \ # SVM model for face recognition 
        -l ./output/le.pickle \ # labels
        -c 0.5 # face threshold
```