import numpy as np
import matplotlib.pyplot as plt
import dataUtils
import model_2020125001 as model

import json
from json import JSONEncoder
import numpy

np.random.seed(1)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# save weights of model from json file
def saveParams(params, path):
    with open(path, "w") as make_file:
        json.dump(params, make_file, cls=NumpyArrayEncoder)
    print("Done writing serialized NumPy array into file")

# load weights of model from json file
def loadParams(path):
    with open(path, "r") as read_file:
        print("Converting JSON encoded data into Numpy array")
        decodedArray = json.load(read_file)
    return decodedArray


def main():
    epochs = 61
    learning_rate = 0.8
    batch_size =128
    decay_rate=1
    resume = False # path of model weights
    model_weights_path = 'weights_2020125001.json'

    ### dataset loading 하기.
    dataPath = 'dataset/train'
    valPath = 'dataset/val'
    dataloader = dataUtils.Dataloader(dataPath, minibatch=batch_size)
    val_dataloader = dataUtils.Dataloader(valPath)
    
    nSample = dataloader.len()
    layerDims = [7500,10,10,1]

    simpleNN = model.NeuralNetwork(layerDims, nSample)
    if resume:
        simpleNN.parameters = loadParams(resume)

    for epoch in range (epochs):     
        training(dataloader, simpleNN, learning_rate, epoch)
        if epoch%10==1:
            
            
            learning_rate=learning_rate/(1+decay_rate*epoch)
#            learning_rate=learning_rate * (0.95)**epoch
#            learning_rate=1/np.sqrt(epoch) * learning_rate
            
            print(epoch, "       " , learning_rate)
            validation(val_dataloader,simpleNN)
            
    validation(val_dataloader,simpleNN)
    saveParams(simpleNN.parameters, model_weights_path)


def validation(dataloader, simpleNN):
    for i, (images, targets) in enumerate(dataloader):
        # do validation
        A4=simpleNN.forward(images)
        predictions = simpleNN.predict(images)
        print(predictions)
        print(targets)
        Y_train=targets
        cost=simpleNN.compute_cost(A4,targets)
        print ('Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')
        print ("Cost after epoch: %f" %(cost))
        pass
def training(dataloader, simpleNN, learning_rate, epoch):

    for i, (images, targets) in enumerate(dataloader):
        # do training
        A4=simpleNN.forward(images)
        cost=simpleNN.compute_cost(A4,targets)
        simpleNN.backward()
        simpleNN.update_params(learning_rate)
        #print ('Accuracy: %d' % float((np.dot(targets,predictions.T) + np.dot(1-targets,1-predictions.T))/float(targets.size)*100) + '%')        
        pass

    return simpleNN
 

if __name__=='__main__':
    main()