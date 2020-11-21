import numpy as np

def sigmoid(x):
    """
    sigmoid 함수

    Arguments:
        x:  scalar 또는 numpy array

    Return:
        s:  sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))

    return s

def relu(x):
    """
    ReLU 함수

    Arguments:
        x : scalar 또는 numpy array

    Return:
        s : relu(x)
    """
    s = np.maximum(0,x)
    
    return s

class NeuralNetwork:
    def __init__(self,layerDims, nSample):
        '''
        학습할 네트워크.

        Arguments:
            layerDims [array]: layerDims[i] 는 레이어 i의 hidden Unit의 개수 (layer0 = input layer)
            nSample: 데이터셋의 샘플 수
        '''

        self.nSample = nSample
        self.nlayer = len(layerDims)-1

        self.parameters = self.weightInit(layerDims)
        self.grads = {}
        self.vel = {}
        self.s = {}
        self.cache = {}
        self.initialize_optimizer()

    def weightInit(self, layerDims):

        np.random.seed(1)
        parameters = {}

        for l in range(1, len(layerDims)):
            parameters['W' + str(l)] = np.random.randn(layerDims[l],layerDims[l-1]) *0.01
            parameters['b' + str(l)] = np.zeros((layerDims[l],1))

        return parameters

    # iniitialize parameter for optimizer
    def initialize_optimizer(self):

        parameters=self.parameters
        v={}
        s={}
        
        for l in range(self.nlayer):
            v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
            s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

        self.vel=v
        self.s=s
        
    def forward(self, X):
        '''
        forward propagation

        Arguments:
            X: input data

        Return:
            A23: network output
        '''
#        self.cache['A0']=X
#        for l in range(self.nlayer):
#            cache["Z1"+str(l+1)]=np.dot(parameters["W"+str(l+1),cache["A"+str(l)]])
#            cache["A1"+str(l+1)]=

        ## 코딩시작 
        drop=0.5
        W1,b1,W2,b2,W3,b3 = self.parameters['W1'],self.parameters['b1'],self.parameters['W2'],self.parameters['b2'],self.parameters['W3'],self.parameters['b3']
#        W4,b4=self.parameters['W4'],self.parameters['b4']
        
        
        Z1= np.dot(W1,X) +b1
        A1= relu(Z1)
#        D1= np.random.rand(A1.shape[0],A1.shape[1])
#        D1= drop < D1
#        A1= np.multiply(A1,D1)
#        A1=A1/drop
        
        Z2= np.dot(W2,A1)+ b2
        A2= relu(Z2)
        D2= np.random.rand(A2.shape[0],A2.shape[1])
        D2= drop < D2
        A2= np.multiply(A2,D2)
        A2= A2 / drop
        
        Z3= np.dot(W3,A2)+ b3
        A3= sigmoid(Z3)
#        D3= np.random.rand(A3.shape[0],A3.shape[1])
#        D3=drop < D3
#        A3= np.multiply(A3,D3)
#        A3=A3/drop
        
#        Z4=np.dot(W4,A3)+b4
#        A4=sigmoid(Z4)
        
        self.cache['X']=X
        self.cache['A1']=A1
        self.cache['A2']=A2
        self.cache['A3']=A3
#        self.cache['A4']=A4
        self.cache['Z1']=Z1
        self.cache['Z2']=Z2
        self.cache['Z3']=Z3
#        self.cache['Z4']=Z4


        return A3

    def backward(self,lambd=0.7):
        '''
        regularization term이 추가된 backward propagation.

        Arguments:
            lambd

        Return:
        '''

        X=self.cache['X']
        A1,A2,A3=self.cache['A1'],self.cache['A2'],self.cache['A3']
        Y=self.cache['Y']
        W3,b3,W2,b2,W1,b1=self.parameters['W3'],self.parameters['b3'],self.parameters['W2'],self.parameters['b2'],self.parameters['W1'],self.parameters['b1']

#        A4,W4,b4=self.cache['A4'],self.parameters['W4'],self.parameters['b4']

        m=X.shape[1]

#        dZ4=A4 - Y ## y 값을 불러와야된다.
#        dW4=1/m* np.dot(dZ4,np.transpose(A3))+ lambd * W4 / m
#        db4=1/m*np.sum(dZ4,axis=1,keepdims=True)

#        dA3=np.dot(np.transpose(W4),dZ4)
#        ## 이부분 주목
#        dZ3= np.dot(np.transpose(W4),dZ4)*np.int64(A3>0)#g[l]`(z[l])
#        dW3=1/m* np.dot(dZ3,np.transpose(A2))+ lambd * W3 / m
#        db3=1/m* np.sum(dZ3,axis=1,keepdims=True)

        dZ3=A3 - Y ## y 값을 불러와야된다.
        dW3=1/m* np.dot(dZ3,np.transpose(A2))+ lambd * W3 / m
        db3=1/m*np.sum(dZ3,axis=1,keepdims=True)
        
        
        dA2=np.dot(np.transpose(W3),dZ3)
        ## 이부분 주목
        dZ2= np.dot(np.transpose(W3),dZ3)*np.int64(A2>0)#g[l]`(z[l])
        dW2=1/m* np.dot(dZ2,np.transpose(A1))+ lambd * W2 / m
        db2=1/m* np.sum(dZ2,axis=1,keepdims=True)
        
        dA1=np.dot(np.transpose(W2),dZ2)
        
        dZ1= np.dot(np.transpose(W2),dZ2)*np.int64(A1>0)#g[l]`(z[l])
        dW1=1/m* np.dot(dZ1,np.transpose(X))+ lambd * W1 / m
        db1=1/m*np.sum(dZ1,axis=1,keepdims=True)


        self.grads.update(dW1=dW1, db1= db1, dW2=dW2, db2=db2,dW3=dW3,db3=db3)
#        self.grads.update(dW4=dW4, db4=db4)




        return


    def compute_cost(self, A3, Y, lambd=0.7):
 

        epsilon=1e-8
        self.cache.update(Y=Y)
        W1, W2, W3 = self.parameters['W1'], self.parameters['W2'], self.parameters['W3']
#        W4=self.parameters['W4']
        
        m=Y.shape[0]
        
        logprobs=-(np.multiply(np.log(A3+epsilon),Y) + np.multiply(np.log(1 - A3 +epsilon),1 - Y))
        
        reg=lambd / (2 * m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
#        reg=0
        
        cost = np.sum(logprobs) * (1/m) + reg

        
        cost = float(np.squeeze(cost))  

        assert(isinstance(cost, float))
        
        return cost

    def update_params(self, learning_rate=1.2, beta2=0.999, epsilon=1e-8):
        '''
        backpropagation을 통해 얻은 gradients를 update한다.

        Arguments:
            learning_rate:  학습할 learning rate

        Return:
        '''

        parameters=self.parameters
        grads=self.grads
        L=self.nlayer
        v=self.vel
        s=self.s
        beta1=0.9
        v_corrected={}
        s_corrected={}
        for l in range(L):
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
    
#            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1)
#            v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1)

#            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.multiply(grads["dW"+str(l+1)],grads["dW" + str(l+1)])
#            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.multiply(grads["db" + str(l+1)],grads["db" + str(l+1)])
#
#            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2)
#            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2)

        ##gradiant descent
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] -learning_rate * v["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] -learning_rate * v["db" + str(l+1)]

#            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
#            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
#
#            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" +str(l+1)] / np.sqrt(s["dW" + str(l+1)]+epsilon)
#            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" +str(l+1)] / np.sqrt(s["db" + str(l+1)]+epsilon)




#            W1,b1,W2,b2,W3,b3 = self.parameters['W1'],self.parameters['b1'],self.parameters['W2'],self.parameters['b2'],self.parameters['W3'],self.parameters['b3']
#            dW1,db1,dW2,db2,dW3,db3 = self.grads['dW1'],self.grads['db1'],self.grads['dW2'],self.grads['db2'],self.grads['dW3'],self.grads['db3']
##            W4,dW4,b4,db4=self.parameters['W4'],self.grads['dW4'],self.parameters['b4'],self.grads['db4']
#            
#            a=learning_rate
#            W1 = W1-a*dW1/ np.sqrt(s['dW1']+epsilon)
#            b1 = b1-a*db1/ np.sqrt(s['db1']+epsilon)
#            W2 = W2-a*dW2/ np.sqrt(s['dW2']+epsilon)
#            b2 = b2-a*db2/ np.sqrt(s['db2']+epsilon)
#            W3 = W3-a*dW3/ np.sqrt(s["dW3"]+epsilon)
#            b3 = b3-a*db3/ np.sqrt(s["db3"]+epsilon)
#            W4 = W4-a*dW4/ np.sqrt(s["dW4"]+epsilon)
#            b4 - b4-a*db4/ np.sqrt(s["db4"]+epsilon)
            
            
        
        self.parameters=parameters


        return 

    def predict(self,X):
        '''
        학습한 network가 잘 학습했는지, test set을 통해 확인한다.

        Arguments:
            X: input data
        Return:
        '''


        A3=self.forward(X)
        predictions=np.where(A3 >0.5,1,0)

        return predictions
