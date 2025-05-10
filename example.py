import abovo
from abovo import Sequential, DenseLayer, Matrix, ActivationType, LossType

X = Matrix(4, 2)
X[0, 0] = 0; X[0, 1] = 0
X[1, 0] = 0; X[1, 1] = 1
X[2, 0] = 1; X[2, 1] = 0
X[3, 0] = 1; X[3, 1] = 1

y = Matrix(4, 1)
y[0, 0] = 0
y[1, 0] = 1
y[2, 0] = 1
y[3, 0] = 0

model = Sequential()
model.add(DenseLayer(2, 4, ActivationType.RELU))
model.add(DenseLayer(4, 1, ActivationType.SIGMOID))
model.train(X, y, epochs=100, batch_size=1, learning_rate=0.1, loss_type=LossType.MSE)