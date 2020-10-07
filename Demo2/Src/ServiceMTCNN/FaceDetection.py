import detect_face as lib
import tensorflow as tf
import cv2
import base64
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from pprint import pprint

def detectFace(img, sess, pnet, rnet, onet):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # sess = tf.Session()
    # pnet, rnet, onet = lib.create_mtcnn(sess, None)
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    boxes, _ = lib.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    faces = []
    for box in boxes:
        tmp = []
        for i in range(4):
            tmp.append(int(box[i]))
        cv2.rectangle(img, (tmp[0], tmp[1]), (tmp[2], tmp[2]), (255, 0, 0), 2)
        faces.append({'x': tmp[0], 'y': tmp[1], 'width': (tmp[2] - tmp[0]), 'height': (tmp[3] - tmp[1])})

    # sess.close()
    # del sess, pnet, rnet, onet, minsize, threshold, factor, boxes
    del pnet, rnet, onet, minsize, threshold, factor, boxes
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return faces

class MTCNN:
    def __init__(self, _imagePath, _dirResultFile):
        self.imagePath = _imagePath
        self.dirResultFile = _dirResultFile

    def detect(self, sess, pnet, rnet, onet):
        fileName, fileExt = os.path.splitext(self.imagePath)
        image = cv2.imread(self.imagePath)

        print("This is image path")
        print(self.imagePath)

        print('Face detection uses Multi-Task Convolutional Neural Network')

        faces = detectFace(image, sess, pnet, rnet, onet)
        for face in faces:
            x = face['x']
            y = face['y']
            w = face['width'] + x
            h = face['height'] + y
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

        resultFile = fileName.split('/')[-1] + fileExt
        print('RESULT FILE:', resultFile)
        cv2.imwrite(os.path.join(self.dirResultFile, resultFile), image)
        #cv2.imwrite('./result
        #.jpg', image)
        result = {}
        #result['base64'] = base64.b64encode(open(self.dirResultFile + 'result.jpg', 'rb').read())
        result['fileName'] = resultFile
        result['range'] = faces

        return result
        
    def detect_folder(self, folder, fromf, tof):
        files = [f for f in listdir(join(fromf, folder)) if isfile(join(fromf, folder, f))]
        bb = []
        sess = tf.Session()
        pnet_fun, rnet_fun, onet_fun = lib.create_mtcnn(sess, None)
        for f in files:
            self.imagePath = join(fromf, folder, f)
            result = self.detect(sess, pnet_fun, rnet_fun, onet_fun)
            pprint(result)
            idx = np.argmax([re['width'] * re['height'] for re in result['range']])
            bb.append([result['range'][idx]])
        np.save(join(tof, folder, 'bb.npy'), bb) 
        np.save(join(tof, folder, 'file_list.npy'), files)
