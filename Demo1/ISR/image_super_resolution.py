# Super resolution
import numpy as np
from PIL import Image
from ISR.models import RDN
from matplotlib import image
from matplotlib import pyplot

img = Image.open('adrian.jpg')
lr_img = np.array(img)
print(lr_img.shape)

rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')

sr_img = rdn.predict(lr_img)
output = Image.fromarray(sr_img)

output.save('adrian_highres.jpg')
pyplot.imshow(output)
pyplot.show()