import definitions
import numpy as np
import copy as cp
from logistic_func import getF, getFPrime


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

    layerCnt = len( neuronCntList )
    # print(layerCnt)


    assert neuronCntList[0]==xTrain.shape[1]
    featureCnt = neuronCntList[0]

    assert neuronCntList[-1]==yTrain.shape[1]
    outputVectorSize = neuronCntList[-1]

    assert xTrain.shape[0]==yTrain.shape[0]
    exampleCnt = xTrain.shape[0]

    wAr = getW(neuronCntList, np.random.random)
    # print(wAr)

    for roundIdx in range(totalRound):
        errorInThisRound = 0.0

        delW = getW(neuronCntList=neuronCntList, arrayGettingMethod=np.zeros)
        # print( delW )

        for exampleIdx in range(exampleCnt):
            curX = xTrain[exampleIdx,:]
            # print("curX")
            # print(curX)

            curY = yTrain[exampleIdx,:]
            # print("curY")
            # print(curY)

            vAr = getZigzagged2dArray(neuronCntList, np.zeros)
            yAr = getZigzagged2dArray(neuronCntList, np.zeros)

            # start of forward propagation algorithm
            yAr[0] = curX
            # print(" yAr[0] ")
            # print(yAr[0])


            forwardProp(wAr=wAr, yAr=yAr, vAr=vAr)
            estimatedY = yAr[ -1 ]

            errorForThisExample = distSq(estimatedY, curY)
            errorInThisRound += errorForThisExample

            eAr = getZigzagged2dArray(neuronCntList, np.zeros)
            delAr = getZigzagged2dArray(neuronCntList, np.zeros)


            # start of back propagation
            eAr[-1] = getF( vAr[-1] ) - curY
            # print("eAr[-1]")
            # print(eAr[-1])

            delAr[-1] = getFPrime( vAr[-1] ) * eAr[-1]
            # print("delAr[-1]")
            # print(delAr[-1])

            backPropagation(wAr=wAr, delAr=delAr, eAr=eAr, yAr=yAr, vAr=vAr)


            # combining forward and backward propagation
            for layerIdx in range(1,layerCnt):
                prevYWithBias = np.append( 1, yAr[layerIdx-1] )
                prevYWithBias = prevYWithBias[np.newaxis]

                delWThisLayerThisExample = np.multiply(delAr[layerIdx], prevYWithBias.T)
                # print( delWThisLayerThisExample.shape )

                delW[layerIdx] = delW[layerIdx] + delWThisLayerThisExample.T


        for layerIdx in range(1,layerCnt):
            wAr[layerIdx] = wAr[layerIdx] - learningRate * delW[layerIdx]



        print("roundIdx")
        print(roundIdx)

        print("errorInThisRound")
        print(errorInThisRound)






def getW(neuronCntList, arrayGettingMethod):

    w = []
    w.append(None)  # first layer is input layer
                    # So no weight vector is defined for that
    for i in range(1, len(neuronCntList) ):
        prevLayerNeuronCnt = neuronCntList[i - 1]
        curLayerNeuronCnt = neuronCntList[i]

        wr = arrayGettingMethod((curLayerNeuronCnt, prevLayerNeuronCnt+1))
        w.append(wr)

    return w


def getZigzagged2dArray( columnCntList, arrayGettingMethod ):
    za = []
    for i in columnCntList:
        addee = arrayGettingMethod(i)
        za.append(addee)
    return za



def forwardProp(wAr, yAr, vAr):
    layerCnt =  len(wAr)
    for layerIdx in range(1, layerCnt):
        # print("layerIdx")
        # print(layerIdx)

        curLayerW = wAr[layerIdx]
        # print("curLayerW")
        # print(curLayerW)

        prevYWithBias = np.append(1, yAr[layerIdx - 1])
        # print("prevYWithBias")
        # print(prevYWithBias)

        vAr[layerIdx] = np.dot(curLayerW, prevYWithBias)
        # print("vAr[layerIdx]")
        # print(vAr[layerIdx])

        yAr[layerIdx] = getF(x=vAr[layerIdx])

        # print("yAr[layerIdx]")
        # print(yAr[layerIdx])


def distSq(x, y):
    assert x.shape == y.shape
    return np.sum((x-y)**2)


def backPropagation(wAr, yAr, delAr, eAr, vAr):
    layerCnt = len(wAr)
    for layerIdx in range(layerCnt-2, -1, -1):
        # print("layerIdx")
        # print(layerIdx)

        nextLayerW = wAr[layerIdx+1][:,1:]
        # print("nextLayerW")
        # print(nextLayerW)

        nextLayerDel = delAr[layerIdx+1]
        # print("nextLayerDel")
        # print(nextLayerDel)

        eAr[layerIdx] = np.dot(nextLayerW.T, nextLayerDel)
        # print( "eAr[layerIdx]" )
        # print(eAr[layerIdx])

        delAr[layerIdx] = eAr[layerIdx] * getFPrime( vAr[layerIdx] )

pathToDataFileFromProjectRoot = "data1/data.txt";
dataSet = getDataSet(pathToFileFromProjectRoot=pathToDataFileFromProjectRoot)
ret = getXY(dataSet=dataSet, featureCnt=2, outputVecSize=2)

X = ret['X']
Y = ret['Y']

assert X.shape[0] == Y.shape[0] # Both X and Y should contain same number of examples
exampleCnt = X.shape[0]


# bias = np.ones( (exampleCnt, 1) )
# X = np.append(bias, X, axis=1)

# print(X)
# print(Y)

featureCnt = X.shape[1]
outputVecSize = Y.shape[1]


neuronCntList=[featureCnt, 5, 4, 3, outputVecSize]
learningRate = 0.001
roundCnt = 10000
getNN(xTrain=X, yTrain=Y, neuronCntList=neuronCntList, learningRate=learningRate,
      totalRound=roundCnt)