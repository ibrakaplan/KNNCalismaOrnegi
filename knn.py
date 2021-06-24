import pandas as pd
import matplotlib.pyplot as plt

k = pd.read_csv("Position_Salaries.csv")
print("islemin sonucu :", k.head())

"""
Leave-One-Out (LOO)
Her seferinde bir noktayi test sete ayir.
"""
from sklearn import neighbors, model_selection, metrics

X = k[["Level"]]
y = k["Salary"]
# print (metrics.SCORERS.keys())

testsonuc = []
trainsonuc = []
for i in range(1,10):
    print ("K :", i)
    model = neighbors.KNeighborsRegressor(n_neighbors=i)
    cv_sonucları = model_selection.cross_validate(
        estimator=model, X=X, y=y, cv=model_selection.LeaveOneOut(), scoring="neg_mean_absolute_error", return_train_score=True)
    k_testsonuc = cv_sonucları["test_score"].mean()
    k_trainsonuc = cv_sonucları["train_score"].mean()
    testsonuc.append(k_testsonuc)
    trainsonuc.append(k_trainsonuc)
    print (k_testsonuc, k_trainsonuc)


plt.plot(range(1,10),testsonuc, label = "testsonuc", color = 'red')
plt.plot(range(1,10),trainsonuc, label = "trainsonuc", color = 'blue')
plt.legend()
plt.show()



# x = list(range(-10,10))
# y = [i**2 for i in x]

# plt.plot(x, y, color='red', marker='o')
# plt.grid()
# plt.savefig("poly.png")
# plt.close()
# plt.show()
