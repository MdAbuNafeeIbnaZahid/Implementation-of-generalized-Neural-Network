import definitions
import numpy as np


def getDataSet( pathToFileFromProjectRoot ):
    pathToFile = definitions.ROOT_DIR + "/" + pathToFileFromProjectRoot
    openedFile = open(pathToFile, "rb")

    loadedText = np.loadtxt(openedFile, skiprows=0, dtype=str)
    floatDataSet = loadedText.astype(np.float)
    return floatDataSet


def getNN(trainset, neuronCntList):

    featureCnt = neuronCntList[0]
    outputVectorSize = neuronCntList[-1]

    w = getRandomW(neuronCntList)


def getRandomW(neuronCntList):
    w = []
    w.append(None)  # first layer is input layer
                    # So no weight vector is defined for that
    for i in range(1, neuronCntList.length):
        prevLayerNeuronCnt = neuronCntList[i - 1]
        curLayerNeuronCnt = neuronCntList[i]

        wr = np.random.random((curLayerNeuronCnt, prevLayerNeuronCnt))
        w.append(wr)

    return wr




pathToDataFileFromProjectRoot = "data1/data.txt";
dataSet = getDataSet(pathToFileFromProjectRoot=pathToDataFileFromProjectRoot)
print( dataSet )