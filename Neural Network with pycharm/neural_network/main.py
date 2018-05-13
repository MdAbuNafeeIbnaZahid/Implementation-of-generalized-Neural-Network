import definitions
import numpy as np


def getDataSet( pathToFileFromProjectRoot ):
    pathToFile = definitions.ROOT_DIR + "/" + pathToFileFromProjectRoot
    openedFile = open(pathToFile, "rb")

    loadedText = np.loadtxt(openedFile, skiprows=0, dtype=str)
    floatDataSet = loadedText.astype(np.float)
    return floatDataSet

def getXY( dataSet, featureCnt, outputVecSize ):
    # print(dataSet)
    exampleCnt = dataSet.shape[0]

    X =  dataSet[:, 0:featureCnt ]
    Y = dataSet[:,-1]

    yVec = np.zeros( (exampleCnt, outputVecSize) )
    for i in range(exampleCnt):
        colToMake1 = int(Y[i])-1
        yVec[ i, colToMake1 ] = 1

    Y = yVec

    ret = {}
    ret['X'] = X
    ret['Y'] = Y

    return ret;


def getNN(xTrain, yTrain, neuronCntList, learningRate, totalRound):

    assert neuronCntList[0]==xTrain.shape[1]
    featureCnt = neuronCntList[0]

    assert neuronCntList[-1]==yTrain.shape[1]
    outputVectorSize = neuronCntList[-1]

    assert xTrain.shape[0]==yTrain.shape[0]
    exampleCnt = xTrain.shape[0]

    w = getRandomW(neuronCntList)
    print(w)

    for roundIdx in range(totalRound):
        for exampleIdx in range(exampleCnt):
            pass


def getRandomW(neuronCntList):

    w = []
    w.append(None)  # first layer is input layer
                    # So no weight vector is defined for that
    for i in range(1, len(neuronCntList) ):
        prevLayerNeuronCnt = neuronCntList[i - 1]
        curLayerNeuronCnt = neuronCntList[i]

        wr = np.random.random((curLayerNeuronCnt, prevLayerNeuronCnt+1))
        w.append(wr)

    return w




pathToDataFileFromProjectRoot = "data1/data.txt";
dataSet = getDataSet(pathToFileFromProjectRoot=pathToDataFileFromProjectRoot)
ret = getXY(dataSet=dataSet, featureCnt=2, outputVecSize=2)

X = ret['X']
Y = ret['Y']

assert X.shape[0] == Y.shape[0] # Both X and Y should contain same number of examples
exampleCnt = X.shape[0]


bias = np.ones( (exampleCnt, 1) )
X = np.append(bias, X, axis=1)

print(X)
print(Y)

featureCnt = X.shape[1]
outputVecSize = Y.shape[1]


neuronCntList=[featureCnt, 5, 4, outputVecSize]
learningRate = 0.1
roundCnt = 100
getNN(xTrain=X, yTrain=Y, neuronCntList=neuronCntList, learningRate=learningRate, totalRound=roundCnt)