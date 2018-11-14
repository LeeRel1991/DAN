# coding=utf-8

## demo for images of a dir

import tensorflow as tf
from ImageServer import ImageServer
from models import DAN
import numpy as np
import cv2, glob, time
import os

# initLandmarks saved from testdata
initLandmarks = np.load("../data/initLandmarks.npy")
dan = DAN(initLandmarks)

img_dir = "../data/imgs"
files = glob.glob(os.path.join(img_dir, "*.jpg"))

with tf.Session() as sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", sess.graph)

    Saver.restore(sess, './Model/Model')
    print('Pre-trained model has been loaded!')

    errs = []
    for filename in files:
        img = cv2.imread(filename)
        # preprocess, to gray, resize, normalization etc.
        gray = cv2.resize(img, (112, 112))
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        x = gray.astype(np.float32)
        mu = np.mean(x)
        std = np.std(x)
        x = (x-mu)/std
        x = x[np.newaxis, :,:, np.newaxis]

        tic = time.time()

        kpts = sess.run(dan['S2_Ret'], feed_dict={dan['InputImage']: x,
                                                    dan['S1_isTrain']: False,
                                                    dan['S2_isTrain']: False})

        print("time ", time.time() - tic)

        # reproject to original img
        kpts = kpts/112 * img.shape[0]

        # draw and display
        for s,t in kpts.reshape((-1, 2)):
            img = cv2.circle(img, (int(s), int(t)), 1, (0,0,255), 2)
        cv2.imshow("out", img)
        cv2.waitKey(200)

