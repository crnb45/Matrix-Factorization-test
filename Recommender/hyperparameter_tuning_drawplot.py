import numpy as np
import matplotlib.pyplot as plt

reg_data = open("hyperparameter_tuning_kfold.csv")
uxp_train = np.loadtxt(reg_data, delimiter=",")

lnlambda = np.log(uxp_train[:, 0])
train_error = uxp_train[:, 1]
val_error = uxp_train[:, 2]

plt.plot(lnlambda, train_error, marker='.')
plt.plot(lnlambda, val_error, marker='.')
plt.legend(['train err', 'val err'])
plt.ylabel('RMSE')
plt.xlabel('ln λ')
plt.title('Matrix Factorization Lambda Penalty Optimization')
plt.show()