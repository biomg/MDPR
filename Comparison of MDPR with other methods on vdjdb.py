import matplotlib.pyplot as plt
import numpy as np

results_method1_datasetA = np.array([0.752,0.754,0.753,0.749,0.752,0.754,0.755,0.752,0.756,0.753])
results_method2_datasetA = np.array([0.725,0.728,0.723,0.722,0.725,0.726,0.725,0.724,0.723,0.727])
results_method3_datasetA = np.array([0.698,0.705,0.704,0.698,0.703,0.702,0.704,0.705,0.703,0.700])
results_method4_datasetA = np.array([0.686,0.689,0.682,0.685,0.683,0.685,0.687,0.686,0.689,0.682])
results_method5_datasetA = np.array([0.664,0.662,0.666,0.664,0.663,0.664,0.665,0.667,0.663,0.664])
results_method6_datasetA = np.array([0.686,0.684,0.689,0.683,0.684,0.686,0.687,0.683,0.687,0.688])





plt.figure(figsize=(5, 5))


plt.subplot(1, 1, 1)
plt.boxplot([results_method1_datasetA,results_method2_datasetA,results_method3_datasetA,
             results_method4_datasetA, results_method5_datasetA, results_method6_datasetA])
plt.title('VDJdb')
plt.xticks(np.arange(1, 7), ['MDPR', 'SEFEM', 'SPFEM', 'DeepLion', 'DeepCat', 'AttenCaIdX'])
plt.grid(True)
plt.ylim(0.660, 0.760)



plt.tight_layout()

plt.show()
