from assignment1 import load_dataset, train_model, evaluate_model
import matplotlib.pyplot as plt

########################################################################
#                                 Test 2                               #
#                       Run with different epochs                      #
########################################################################

X_train, y_train, X_test, y_test = load_dataset('creditcard.csv')

accuracy_scores=[]

epochs=[5,10,15,20,30]

for i in epochs:
    print('Starting loop for epochs=',i)
    hparams={'epochs':i}
    model, m_train, s_train = train_model(X_train, y_train,hparams=hparams)
    pred, accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, m_train, s_train)
    print('For epoch=',i,' accuracy=',accuracy)
    accuracy_scores.append(accuracy)

plt.plot(epochs,accuracy_scores)
plt.title("Epochs' Effect on Accuracy ")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('plots/ep-acc.jpg')