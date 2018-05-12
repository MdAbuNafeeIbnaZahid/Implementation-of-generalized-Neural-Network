import numpy as np




def getNN(trainset, neuronCntList):

    featureCnt = neuronCntList[0]
    outputVectorSize = neuronCntList[-1]



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