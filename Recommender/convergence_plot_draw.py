import numpy as np
import matplotlib.pyplot as plt

gcmc_feat = open("gcmc_feat.csv")
gcmc_epoch_train_val = np.loadtxt(gcmc_feat, delimiter=",")
matrix_factorization_train_error = open("train_err_1651448342.csv")
matfac_train = np.loadtxt(matrix_factorization_train_error, delimiter=",")
matrix_factorization_val_error = open("val_err_1651448342.csv")
matfac_val = np.loadtxt(matrix_factorization_val_error, delimiter=",")

epoch = gcmc_epoch_train_val[:, 0]
gcmc_train = gcmc_epoch_train_val[:, 1]
gcmc_val = gcmc_epoch_train_val[:, 2]

plt.plot(epoch, gcmc_train, color='orange', ls='--')
plt.plot(epoch, gcmc_val, 'b--')
plt.plot(epoch, matfac_train, color='orange')
plt.plot(epoch, matfac_val, color='b')
plt.legend(['GC-MC Train Error', 'GC-MC Val Error', 'Matrix Factorization Train Error (λ=0.002)', 'Matrix Factorization Val Error (λ=0.002)'])
plt.ylabel('RMSE')
plt.xlabel('Epochs')
plt.title('Algorithm Convergence and Performance')
plt.show()