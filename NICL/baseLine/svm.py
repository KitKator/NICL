from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


svm_model = SVC(kernel="rbf", probability=True)

svm_model.fit(X_train_A, y_train_A)
y_pred_A = svm_model.predict(X_test_A)
accuracy_A = accuracy_score(y_test_A, y_pred_A)

print_confusion_matrix(y_test_A, y_pred_A)

svm_model.fit(X_train_B, y_train_B)
y_pred_A_B = svm_model.predict(X_test_A)
y_pred_B = svm_model.predict(X_test_B)
accuracy_A_B = accuracy_score(y_test_A, y_pred_A_B)
accuracy_B = accuracy_score(y_test_B, y_pred_B)

svm_model.fit(X_train_C, y_train_C)
y_pred_A_C = svm_model.predict(X_test_A)
y_pred_B_C = svm_model.predict(X_test_B)
y_pred_C = svm_model.predict(X_test_C)
accuracy_A_C = accuracy_score(y_test_A, y_pred_A_C)
accuracy_B_C = accuracy_score(y_test_B, y_pred_B_C)
accuracy_C = accuracy_score(y_test_C, y_pred_C)

