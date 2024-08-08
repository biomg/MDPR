import matplotlib.pyplot as plt
import numpy as np

# 生成随机数据，模拟三个数据集每种方法的十个结果
np.random.seed(10)
results_method1_datasetA = np.array([0.9718,0.9703,0.9713,0.9702,0.9719,0.9718,0.9702,0.9712,0.97109,0.9703])
results_method2_datasetA = np.array([0.9523,0.954,0.9529,0.9549,0.955,0.9515,0.9512,0.9520,0.9524,0.9523])
results_method3_datasetA = np.array([0.908,0.911,0.910,0.907,0.909,0.908,0.910,0.908,0.907,0.908])
results_method4_datasetA = np.array([0.915,0.919,0.914,0.913,0.916,0.920,0.918,0.916,0.913,0.918])
results_method5_datasetA = np.array([0.898,0.903,0.901,0.897,0.896,0.892,0.894,0.897,0.895,0.894])
results_method6_datasetA = np.array([0.888,0.884,0.882,0.884,0.889,0.891,0.884,0.881,0.882,0.884])
print(np.mean(results_method1_datasetA))
print(np.mean(results_method3_datasetA))
results_method1_datasetB = np.array([0.9689,0.9701,0.9692,0.9693,0.9690,0.9703,0.9697,0.96887,0.9701,0.9692])
results_method2_datasetB = np.array([0.9517,0.9518,0.9519,0.9532,0.9520,0.9513,0.9523,0.9530,0.9527,0.9525])
results_method3_datasetB = np.array([0.903,0.9001,0.904,0.905,0.905,0.903,0.902,0.8994,0.903,0.902]) #ok
results_method4_datasetB = np.array([0.922,0.925,0.924,0.921,0.926,0.926,0.924,0.923,0.923,0.925])#ok
results_method5_datasetB = np.array([0.9002,0.901,0.902,0.897,0.896,0.892,0.894,0.897,0.9005,0.9004])
results_method6_datasetB = np.array([0.892,0.889,0.896,0.894,0.894,0.895,0.887,0.895,0.891,0.896]) #ok
print(np.mean(results_method1_datasetB))
results_method1_datasetC = np.random.normal(loc=60, scale=8, size=10)
results_method2_datasetC = np.random.normal(loc=58, scale=9, size=10)
results_method3_datasetC = np.random.normal(loc=62, scale=7, size=10)
results_method4_datasetC = np.random.normal(loc=63, scale=10, size=10)
results_method5_datasetC = np.random.normal(loc=59, scale=11, size=10)
results_method6_datasetC = np.array([0.892,0.889,0.896,0.894,0.894,0.895,0.887,0.895,0.891,0.896])

# 绘制箱型图
plt.figure(figsize=(10, 5))

# 数据集A的箱型图
plt.subplot(1, 2, 1)
plt.boxplot([results_method1_datasetA, results_method2_datasetA, results_method3_datasetA,
             results_method4_datasetA, results_method5_datasetA, results_method6_datasetA])
plt.title('Lan et al.\'s dataset')
plt.xticks(np.arange(1, 7), ['MDPR', 'SEFEM', 'SPFEM', 'DeepLion', 'DeepCat', 'AttenCaIdX'])
plt.grid(True)
plt.ylim(0.880, 0.975)

# 数据集B的箱型图
plt.subplot(1, 2, 2)
plt.boxplot([results_method1_datasetB, results_method2_datasetB, results_method3_datasetB,
             results_method4_datasetB, results_method5_datasetB, results_method6_datasetB])
plt.title('Li et al.\'s dataset')
plt.xticks(np.arange(1, 7), ['MDPR', 'SEFEM', 'SPFEM', 'DeepLion', 'DeepCat', 'AttenCaIdX'])
plt.grid(True)
plt.ylim(0.880, 0.975)
# 数据集C的箱型图

plt.tight_layout()

plt.show()
