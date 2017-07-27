from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.optimizers as Optimizers
import numpy as np
from random import randint
from sys import argv
import h5py

def print_prediction(prediction):
    for i in range(0, len(prediction)):
        output = 'Prediction ' + str(i+1) + ': '
        for j in range(0, len(prediction[i])):
            output += str(prediction[i][j]) + ' '
        print(output)

def normalize(value, valueMin, valueMax, resultMin, resultMax):
    if valueMax-valueMin == 0:
        return 0
    else:
        return (value-valueMin)/(valueMax-valueMin)*(resultMax-resultMin) + resultMin

def getBounds(arr, dim):
    bounds = np.zeros((dim, 2))
    for i in range(0, dim):
        bounds[i][0] = min(min(x[i:]) for x in arr)
        bounds[i][1] = max(max(x[i:]) for x in arr)        
    return bounds

def normalizeDataWithoutBounds(data, dim, min, max):
    data_ = np.copy(data)
    bounds = getBounds(data_, dim)
    for i in range(0, len(data_)):
        for j in range(0, dim):
            data_[i][j] = normalize(data_[i][j], bounds[j][0], bounds[j][1], min, max)
    return data_

def normalizeDataWithBounds(data, dim, bounds, min, max):
    data_ = np.copy(data)
    for i in range(0, len(data_)):
        for j in range(0, dim):
            data_[i][j] = normalize(data_[i][j], bounds[j][0], bounds[j][1], min, max)
    return data_

def renormalizeData(data, dim, bounds):
    data_ = np.copy(data)
    for i in range(0, len(data_)):
        for j in range(0, dim):
            data_[i][j] = normalize(data_[i][j], -1, 1, bounds[j][0], bounds[j][1])
    return data_

def query(count, x, y):
    samples = len(x)
    for i in range (0, count):
        print('---TEST #' + str(i+1) + '---')
        index = randint(0, samples)
        print('Sample: ' + str(index))
        print('Query: ' + str(x[index]))
        print('True: ' + str(y[index]))
        prediction = model.predict(np.array([x[index]]))[0]
        print('Prediction: ' + str(prediction))

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()

# Create training data
#data = np.loadtxt('/media/sebastian/7aed0e14-7811-4a26-99bd-11184b14102a/Development/Theano/pa10_1K.csv') # 1000 random IK training
#data = np.loadtxt('pa10_10K.csv') # 10000 random IK training
#dataTrain = np.loadtxt('pa10_1K.csv')
#dataTest = np.loadtxt('pa10_500.csv')
dataTrain = np.loadtxt('pa10_config000_50k.csv')
dataTest = np.loadtxt('pa10_config000_10k.csv')

trainSamples = 1000
testSamples = 500
dimX = 7
dimY = 6

bounds = np.array([
        [-170.0, 170.0],
        [-60.0, 120.0],
        [-100.0, 150.0],
        [-150.0, 150.0],
        [-95.0, 95.0],
        [-150.0, 150.0],
    ])
bounds *= 3.141592653589793 / 180.0

X = np.zeros((trainSamples,dimX))
Y = np.zeros((trainSamples,dimY))
for i in range(0, trainSamples):
    X[i][0] = dataTrain[i][7]
    X[i][1] = dataTrain[i][8]
    X[i][2] = dataTrain[i][9]
    X[i][3] = dataTrain[i][10]
    X[i][4] = dataTrain[i][11]
    X[i][5] = dataTrain[i][12]
    X[i][6] = dataTrain[i][13]

    Y[i][0] = dataTrain[i][1]
    Y[i][1] = dataTrain[i][2]
    Y[i][2] = dataTrain[i][3]
    Y[i][3] = dataTrain[i][4]
    Y[i][4] = dataTrain[i][5]
    Y[i][5] = dataTrain[i][6]

#X = normalizeDataWithoutBounds(X, dimX, -1, 1)
Y = normalizeDataWithBounds(Y, dimY, bounds, -1, 1)

Xtest = np.zeros((testSamples,dimX))
Ytest = np.zeros((testSamples,dimY))
for i in range(0, testSamples):
    Xtest[i][0] = dataTest[i][7]
    Xtest[i][1] = dataTest[i][8]
    Xtest[i][2] = dataTest[i][9]
    Xtest[i][3] = dataTest[i][10]
    Xtest[i][4] = dataTest[i][11]
    Xtest[i][5] = dataTest[i][12]
    Xtest[i][6] = dataTest[i][13]

    Ytest[i][0] = dataTest[i][1]
    Ytest[i][1] = dataTest[i][2]
    Ytest[i][2] = dataTest[i][3]
    Ytest[i][3] = dataTest[i][4]
    Ytest[i][4] = dataTest[i][5]
    Ytest[i][5] = dataTest[i][6]

Ytest = normalizeDataWithBounds(Ytest, dimY, bounds, -1, 1)

# Define network
model = Sequential()
model.add(Dense(dimX, input_dim=dimX, init='uniform', bias=False))
model.add(Activation('tanh'))
model.add(Dropout(0.05))
model.add(Dense(150, init='uniform', bias=False))
model.add(Activation('tanh'))
model.add(Dropout(0.05))
model.add(Dense(75, init='uniform', bias=False))
model.add(Activation('tanh'))
model.add(Dropout(0.05))
model.add(Dense(50, init='uniform', bias=False))
model.add(Activation('tanh'))
model.add(Dropout(0.05))
model.add(Dense(25, init='uniform', bias=False))
model.add(Activation('tanh'))
model.add(Dropout(0.05))
model.add(Dense(10, init='uniform', bias=False))
model.add(Activation('tanh'))
model.add(Dropout(0.05))
model.add(Dense(dimY, init='uniform', bias=False))
model.add(Activation('tanh'))

# Generate network
opt = Optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)
model.compile(optimizer=opt, loss='mse')

# Train network
epoch = 0
error = 2
while error > 0.001:
    epoch += 1
    error = model.train_on_batch(X, Y)
    print('==========')
    print('Epoch: ' + str(epoch) + ' Training Error: ' + str(error))
    print('==========')

query(10, X, Y)

#model.save_weights("network")
#print_structure("network")

#weights = model.get_weights()[0]
#print(weights)

#file = open("network", 'w')
#file.write("HELLO WORLD")
#file.close()