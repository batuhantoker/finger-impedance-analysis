import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np
def data_preprocess(X):
    x_new = np.asarray(X[X.columns[0]])
    x_sep = np.asarray([rr.reshape(-1) for rr in x_new])
    for i in range(1,len(X.columns)):
        print(i)
        x_new = np.asarray(X[X.columns[i]])
        x_sep1 = np.asarray([rr.reshape(-1) for rr in x_new])
        x_sep = np.hstack((x_sep, x_sep1))
    return x_sep

data =  pd.read_pickle("./data4s3.pkl").reset_index(drop=True)
#data2 = pd.read_pickle("./data5s1.pkl").reset_index(drop=True)
#data=data.append(data2)
data=data[data.movement_id != 0]

X=data.loc[:, data.columns != 'movement_id']
print(X.columns)
# x_new=np.asarray(X['resized_img_ext'])
# x_sep1 = np.asarray([rr.reshape(-1) for rr in x_new])
# x_new=np.asarray(X['resized_img_flex'])
# x_sep2 = np.asarray([rr.reshape(-1) for rr in x_new])
# x_sep=np.hstack((x_sep1,x_sep2))

# x_new=np.asarray(X['mean_shift_flex'])
# x_sep1=np.asarray([rr.reshape(-1) for rr in x_new])
# x_new=np.asarray(X['mean_shift_ext'])
# x_sep2=np.asarray([rr.reshape(-1) for rr in x_new])
# x_new=np.asarray(X['HoG_flex'])
# x_sep3=np.asarray([rr.reshape(-1) for rr in x_new])
# x_new=np.asarray(X['HoG_ext'])
# x_sep4=np.asarray([rr.reshape(-1) for rr in x_new])
# x_new=np.asarray(X['Cog_flex'])
# x_sep5=np.asarray([rr.reshape(-1) for rr in x_new])
# x_new=np.asarray(X['Cog_ext'])
# x_sep6=np.asarray([rr.reshape(-1) for rr in x_new])
x_new=np.asarray(X['canny_ext'])
x_sep7=np.asarray([rr.reshape(-1) for rr in x_new])
x_new=np.asarray(X['canny_flex'])
x_sep8=np.asarray([rr.reshape(-1) for rr in x_new])
# x_new=np.asarray(X['max_cord_ext'])
# x_sep9=np.asarray([rr.reshape(-1) for rr in x_new])
# x_new=np.asarray(X['max_cord_flex'])
# x_sep10=np.asarray([rr.reshape(-1) for rr in x_new])
x_new=np.asarray(X['Harris_ext'])
x_sep11=np.asarray([rr.reshape(-1) for rr in x_new], dtype=object)
x_new=np.asarray(X['Harris_flex'])
x_sep12=np.asarray([rr.reshape(-1) for rr in x_new], dtype=object)


harris_min=[np.min([rr.shape for rr in x_sep11]),np.min([rr.shape for rr in x_sep12])]
x_sep11=np.asarray([rr[0:np.min(harris_min)] for rr in x_sep11])
x_sep12=np.asarray([rr[0:np.min(harris_min)] for rr in x_sep12])
x_sep=np.hstack((x_sep7,x_sep8,x_sep11,x_sep12)) #
# x_sep=np.hstack((x_sep1,x_sep2,x_sep3,x_sep4,x_sep5,x_sep6,x_sep7,x_sep8,x_sep9,x_sep10,x_sep11,x_sep12)) #
#x_sep=data_preprocess(X)
print(x_sep.shape)

y=np.asarray(data['movement_id']).reshape(-1)
y=y.astype('int')
print(np.unique(y))
print((y.shape))

X_train, X_test, y_train, y_test = train_test_split(x_sep, y, test_size=0.25, random_state=42)

sc = StandardScaler()
x_sep = sc.fit_transform(x_sep)




import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier

Y = y
X = x_sep

number_of_k_fold=10
random_seed=42
outcome = []
model_names = []
# Variables for average classification report
originalclass = []
classification = []
#Make our customer score
def classification_report_with_accuracy_score(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    #classes_list=list(report.keys())
    #classes=report.keys()
    #for i in range(len(classes_list)):
    #    report_class=report.get(classes_list[i],{})
    #    print(type(report_class)) #['precision']
    return accuracy_score(y_true, y_pred) # return accuracy score
models = [('LogReg', LogisticRegression()),
          #('SVM', SVC()),
          ('DecTree', DecisionTreeClassifier()),
          ('KNN', KNeighborsClassifier(n_neighbors=15)),
          ('LinDisc', LinearDiscriminantAnalysis()),
          ('GaussianNB', GaussianNB()),
          ('MLPC', MLPClassifier(activation='relu', solver='adam', max_iter=500)),
          ('RFC',RandomForestClassifier()),
          ('ABC', AdaBoostClassifier())]
for model_name, model in models:
    k_fold_validation = model_selection.KFold(n_splits=number_of_k_fold, random_state=random_seed,shuffle=True)
    results = model_selection.cross_val_score(model, X, Y, cv=k_fold_validation, scoring=make_scorer(classification_report_with_accuracy_score))
    outcome.append(results)
    model_names.append(model_name)
    output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
    print(output_message)
print(classification)
fig = plt.figure()
fig.suptitle('Machine Learning Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome)
plt.ylabel('Accuracy [%]')
ax.set_xticklabels(model_names)
plt.savefig('myimage.png', format='png', dpi=1000)
plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=40)
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=number_of_k_fold)
# mlp = MLPClassifier(hidden_layer_sizes=(8,8,8,8), activation='relu', solver='adam', max_iter=500)
# mlp.fit(X_train,y_train)
#
# predict_train = mlp.predict(X_train)
# predict_test = mlp.predict(X_test)
# from sklearn.metrics import classification_report,confusion_matrix
#
# print(mlp.score(X_test, y_test, sample_weight=None))