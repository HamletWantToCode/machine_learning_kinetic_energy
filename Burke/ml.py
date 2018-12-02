import pickle
import numpy as np
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, meanSquareError, max_abs_error, mean_abs_error

np.random.seed(8)

with open('/Users/hongbinren/Downloads/mnt/project/Burke_paper/quantumX1D', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_data = data[:1001]
test_data = data[1001:]

# N=1
train_n = train_data[train_data[:, 0]==1]
test_n = test_data[test_data[:, 0]==1]
train_X, train_y = train_n[:100, 2:], train_n[:100, 1]
mean_X = np.mean(train_X, axis=0, keepdims=True)
train_X -= mean_X
mean_KE = np.mean(train_y)
train_y -= mean_KE
test_X, test_y = test_n[:, 2:], test_n[:, 1]
test_X -= mean_X
test_y -= mean_KE

gamma = 1.0/(2*18.4207**2)
Lambda = 3.79269*1e-9
kernel = rbfKernel(gamma)
model = KernelRidge(kernel, Lambda)
model.fit(train_X, train_y[:, np.newaxis])
predict_y = model.predict(test_X)

unit_conversion = 627.508474
err1 = meanSquareError(predict_y, test_y)*unit_conversion
err2 = mean_abs_error(predict_y, test_y)*unit_conversion
err3 = max_abs_error(predict_y, test_y)*unit_conversion

print("""
    mean abs error           mean square error            max abs error
         %.2f                      %.2f                      %.2f
        """ %(err1, err2, err3))
