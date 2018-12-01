import pickle
import numpy as np
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel

with open('/home/hongbin/Documents/project/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)

train_X, train_y = data[:100, :-1], data[:100, -1]
mean = np.mean(train_y)
train_y -= mean

test_X, test_y = data[150:, :-1], data[150:, -1]
test_y -= mean

kernel = rbfKernel(0.1)
# kernel = laplaceKernel(0.1)
model = KernelRidge(kernel, 1e-15)
model.fit(train_X, train_y[:, np.newaxis])
predict_y = model.predict(test_X)
err = np.mean((predict_y - test_y)**2)
print(err)

import matplotlib.pyplot as plt
plt.plot(test_y, predict_y, 'bo')
plt.plot(test_y, test_y, 'r')
plt.show()
