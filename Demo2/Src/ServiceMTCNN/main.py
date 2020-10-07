from FaceDetection import *

import time

if __name__ == '__main__':
	detector = MTCNN(None, './result')
	begin = time.time()
	detector.detect_folder('image', 'raw_data', 'face')
	print('Elapsed Time:', time.time() - begin)
