root = 'E:/去噪论文插图/tiny'
from pathlib import Path
import cv2
import numpy as np
dir_names = [i.name for i in Path(root).iterdir() if i.is_dir()]

for dir_name in dir_names:
    img_names = [i for i in Path(root+'/'+dir_name).glob('*.png')]

    noised = cv2.imdecode(np.fromfile(str(img_names[1]), dtype=np.uint8), 1)
    clean = cv2.imdecode(np.fromfile(str(img_names[0]), dtype=np.uint8), 1)
    noise = (noised/255) - (clean/255)
    noise = cv2.normalize(noise, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    # print(noise.shape)
    noise = noise[140:420,180:540]
    noised = noised[140:420, 180:540]
    clean = clean[140:420, 180:540]

    tmp = cv2.vconcat([noised, clean, noise])
    cv2.imencode('.jpg', tmp)[1].tofile(root + '/' + dir_name + '.jpg')

    # cv2.imshow('s',tmp)
    # cv2.waitKey()


    # cv2.imencode('.jpg', clean)[1].tofile(root+'/'+dir_name+'clean.jpg')
    # cv2.imencode('.jpg', noised)[1].tofile(root+'/'+dir_name+'noised.jpg')
    # cv2.imencode('.jpg', noise)[1].tofile(root+'/'+dir_name+'noise.jpg')