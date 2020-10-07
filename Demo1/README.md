> Detailed report: [DEMO-1-Report.pdf](../Documents/DEMO-1-Report.pdf).    


Method:
- Step 0 (Optional): enhance image resolution by ISR [10] before do following steps. 
- Step 1: Detect face in an input image by SSD [1, 3]
- Step 2: Find facial landmarks and align face [5, 6]
- Step 3: Extract 128-d embedding vector for each objects [7,8]
- Step 4: Train SVM on output embedding vectors for dataset 
- Step 5: Do face recognition and output results.

# Folder structure:
`dataset`: training dataset which folder name is name/identity of a object.

`face_detection_model`: stores model configuration and pretrained weights of SSD.

`ISR`: image super resolution module [10]. We didnt integrate it into out source because ISR took long time to produre results. To play with it: 
- Firstly, you need to clone ISR: `git clone https://github.com/idealo/image-super-resolution` and use [image_super_resolution.py](ISR/image_super_resolution.py) script to run.
- [Pretrained model](ISR/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5) are also provided.

`output`: contains some pickle files
- [embeddings.pickle](output/embeddings.pickle): stores embedding images or 128-d feature vectors of training dataset.
- [le.pickle](output/le.pickle): corresponding labels for embedding images.
- [recognizer.pickle](output/recognizer.pickle): stores trained model of SVM for face recognition.

`test_images`: test images for evaluating model

Face detection: [face_detection.py](face_detection.py) for image and [face_detection_video.py](face_detection_video.py) for video-stream.

Facial landmarks and alignment: [facial_landmark.py](facial_landmark.py) with pretrained model `shape_predictor_68_face_landmarks.dat` to extract facial keypoints.

Embedding image: [extract_embeddings.py](extract_embeddings.py) based on pretrained OpenFace model `openface_nn4.small2.v1.t7`.

Training SVM for identifying objects: `train_model.py`. 

Face recognition: [recognize.py](recognize.py) for image and [recognize_face_video.py](recognize_face_video.py) for video.

# Dataset:
<table>
    <tr>
        <th>Object</th>
        <th>#Images</th>
    </tr>
    <tr>
        <td>
            <figure>
                <img src="./dataset/adrian/00000.png" alt="adrian" width="150">
                <figcaption>adrian</figcaption>
            </figure>
        </td>
        <td>
            6
        </td>
    </tr>
    <tr>
        <td>
            <figure>
                <img src="./dataset/taylor/00005.jpg" alt="taylor" width="150">
                <figcaption>taylor</figcaption>
            </figure>
        </td>
        <td>
            10
        </td>
    </tr>
    <tr>
        <td>
            <figure>
                <img src="./dataset/justin_bieber/00001.jpg" alt="justin" width="150" height="">
                <figcaption>justin_bieber</figcaption>
            </figure>
        </td>
        <td>
            10
        </td>
    </tr>
    <tr>
        <td>
            <figure>
                <img src="./dataset/obama/00001.jpg" alt="obama" width="150" height="">
                <figcaption>obama</figcaption>
            </figure>
        </td>
        <td>
            10
        </td>
    </tr>
    <tr>
        <td>
            <figure>
                <img src="./dataset/unknown/ian_malcolm.jpg" alt="ian" width="150" height="">
                <figcaption>unknown</figcaption>
            </figure>
        </td>
        <td>
            6
        </td>
    </tr>
</table>

# Installation
```
pip install -r requirements.txt
```

# Usage:
Face recognition for static image:
```bash
python recognize.py -i test_images/obama_1_hi.jpg \ # path to image 
        -d ./face_detection_model/ \ # path to OpenCV's face detector  
        -m ./openface_nn4.small2.v1.t7 \ # path to OpenCV's face embedding model 
        -r ./output/recognizer.pickle \ # SVM model for face recognition 
        -l ./output/le.pickle \ # labels
        -c 0.5 # face threshold
```
Face recognition for video:
```bash
python recognize.py -i path/to/video 
        -d ./face_detection_model/ \ # path to OpenCV's face detector  
        -m ./openface_nn4.small2.v1.t7 \ # path to OpenCV's face embedding model 
        -r ./output/recognizer.pickle \ # SVM model for face recognition 
        -l ./output/le.pickle \ # labels
        -c 0.5 # face threshold
```

# References:

[1] Blog: Face detection. Adrian. Source: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deeplearning/

[2] Blog: Face recognition using dlib. Adrian. Source:
https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-anddeep-learning/

[3] Blog: OpenCV face recognition. Adrian. Source: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

[4] Dlib library by Davis King. Source: http://dlib.net/

[5] Blog: Facial landmark. Adrian. Source: https://www.learnopencv.com/facemark-faciallandmark-detection-using-opencv/

[6] Face alignment with OpenCV. Source: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

[7] OpenFace library. Source: https://cmusatyalab.github.io/openface/

[8] Face recognition by Adam Geitgey. Source:
https://github.com/ageitgey/face_recognition

[9] Blog: Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning. Adam Geitgey. Source: https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

[10] Image Super-Resolution (ISR). Source: https://github.com/idealo/image-superresolution