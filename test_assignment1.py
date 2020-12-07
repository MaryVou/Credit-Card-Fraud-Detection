from assignment1 import load_dataset, train_model, evaluate_model

X_train, y_train, X_test, y_test = load_dataset('creditcard.csv')

###################################################################
#                    Default Hyperparameters                      #
#                 learning_rate=0.01 & epochs=10                  #
###################################################################

model, m_train, s_train = train_model(X_train, y_train)
pred, accuracy, precision, recall, f1 = evaluate_model(
    model, X_test, y_test, m_train, s_train)

print("On test set with default hyperparameters:")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)

###################################################################
#                   Alternative Hyperparameters                   #
#                 learning_rate=0.001 & epochs=30                 #
###################################################################

hparams = {'learning_rate': 0.001, 'epochs': 30}

model, m_train, s_train = train_model(X_train, y_train, hparams=hparams)
pred, accuracy, precision, recall, f1 = evaluate_model(
    model, X_test, y_test, m_train, s_train)

print("On test set, with learning rate %f and %d epochs:" %
      (hparams['learning_rate'], hparams['epochs']))
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)