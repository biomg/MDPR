
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

x=np.load('Experimental_data/c_xyz_outs.npy',allow_pickle=True)
y=np.load('Experimental_data/c_xyz_label.npy',allow_pickle=True)



fpr,tpr, thresholds = roc_curve(y,x)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr,tpr, color='black',linestyle=':',label='Coordinate-hot encoding (AUC = %0.3f)' % auc)

x=np.load('Experimental_data/esm_outs.npy',allow_pickle=True)
y=np.load('Experimental_data/esm_label.npy',allow_pickle=True)

fpr,tpr, thresholds = roc_curve(y,x)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr,tpr, color='black',linestyle='-',label='ESM-fold_esm (AUC = %0.3f)' % auc)


x=np.load('Experimental_data/pair_outs.npy',allow_pickle=True)
y=np.load('Experimental_data/pair_label.npy',allow_pickle=True)
fpr,tpr, thresholds = roc_curve(y,x)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr,tpr, color='black',linestyle='--',label='Alphafold_evoformer (AUC = %0.3f)' % auc)




plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',size=15)
plt.ylabel('True Positive Rate',size=15)

plt.legend(loc="lower right")

plt.show()
