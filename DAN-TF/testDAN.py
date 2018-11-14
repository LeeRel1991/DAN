# coding=utf-8

##测试部分的代码

import tensorflow as tf
from ImageServer import ImageServer
from models import DAN
import numpy as np

datasetDir = "../data/"
testSet = ImageServer.Load(datasetDir + "challengingSet.npz")


def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks

    return y.reshape((nSamples, nLandmarks * 2))


nSamples = testSet.gtLandmarks.shape[0]
imageHeight = testSet.imgSize[0]
imageWidth = testSet.imgSize[1]
nChannels = testSet.imgs.shape[1]

Xtest = testSet.imgs

Ytest = getLabelsForDataset(testSet)

meanImg = testSet.meanImg
stdDevImg = testSet.stdDevImg
initLandmarks = testSet.initLandmarks[0].reshape((-1))

dan = DAN(initLandmarks)

import time
import cv2
with tf.Session() as sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", sess.graph)

    Saver.restore(sess, './Model/Model')
    print('Pre-trained model has been loaded!')

    # Landmark68Test(MeanShape,ImageMean,ImageStd,sess)
    errs = []

    for iter in range(nSamples):

        x, y = Xtest[iter], Ytest[iter]
        x = x[np.newaxis, :, :, :]
        y = y[np.newaxis, :]

        tic = time.time()
        TestErr, kpts = sess.run([dan['S2_Cost'], dan['S2_Ret']], {dan['InputImage']: x,
                                                                   dan['GroundTruth']: y,
                                                                   dan['S1_isTrain']: False,
                                                                   dan['S2_isTrain']: False})
        errs.append(TestErr)

        print("time ", time.time() - tic)
        img = np.squeeze(x)
        for s,t in kpts.reshape((-1, 2)):
            img = cv2.circle(img, (int(s), int(t)), 1, (0), 2)
        cv2.imshow("out", img)
        cv2.waitKey(200)

        print('The mean error for image {} is: {}'.format(iter, TestErr))
    errs = np.array(errs)
    print('The overall mean error is: {}'.format(np.mean(errs)))
