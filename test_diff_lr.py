from assignment1 import load_dataset, train_model, evaluate_model
import matplotlib.pyplot as plt

########################################################################
#                                 Test 1                               #
#                       Run with different learning_rates              #
########################################################################

X_train, y_train, X_test, y_test = load_dataset('creditcard.csv')

accuracy_scores=[]

learning_rate=[0.001,0.005,0.01,0.05,0.1]

for i in learning_rate:
    print('Starting loop for learning_rate=',i)
    hparams={'learning_rate':i}
    model, m_train, s_train = train_model(X_train, y_train,hparams=hparams)
    pred, accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, m_train, s_train)
    print('For lr=',i,' accuracy=',accuracy)
    accuracy_scores.append(accuracy)

plt.plot(learning_rate,accuracy_scores)
plt.title("Learning_rate's Effect on Accuracy ")
plt.xlabel('learning_rate')
plt.ylabel('accuracy')
plt.savefig('plots/lr-acc.jpg')