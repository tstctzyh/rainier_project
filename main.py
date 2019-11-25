import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# sns.set()

def print_score(clf, X, y, cv=0):

    y_pred = clf.predict(X)
    acc_score = accuracy_score(y, y_pred)
    clf_report = classification_report(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"Results:\n")
    print(f"accuracy score: {acc_score:.4f}\n")
    print(f"Classification Report: \n {clf_report}\n")
    print(f"Confusion Matrix: \n {conf_matrix}\n")

    if cv > 1:
        res = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        print(f"Average Accuracy: \t {np.mean(res):.4f}")
        print(f"Accuracy SD: \t\t {np.std(res):.4f}")

def get_head(data):
	print(data.head())

def visualization(data):
	sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
	plt.show()
	sns.set_style('whitegrid')
	sns.countplot(x='Succeeded', data=data, palette='RdBu_r')
	plt.show()

	sns.countplot(x='Succeeded', hue='Attempted', data=data, palette='RdBu_r')
	plt.show()

def checkSuccess(cols):
    success = cols[0]

    if success > 0:
        return 1
    else:
        return 0

def main():
	weather = pd.read_csv('./Rainier_Weather.csv')
	climbing = pd.read_csv('./climbing_statistics.csv')
	merge_data = weather.merge(climbing,on="Date")
	# get_head(merge_data)
	# visualization(merge_data)
	# print(merge_data.describe())
	merge_data['result'] = merge_data[['Succeeded']].apply(checkSuccess, axis=1)
	# sns.countplot(x='result', data=merge_data, palette='RdBu_r')
	# plt.show()
	pre_merge_data=merge_data.drop(['Route','Success Percentage','Succeeded','Date'], axis=1, inplace=False)
	# get_head(pre_merge_data)
	# print(pre_merge_data.info())

	# Train
	X_train, X_test, y_train, y_test = train_test_split(pre_merge_data.drop('result',axis=1),pre_merge_data['result'], test_size=0.30,random_state=11)
	rf_clf = RandomForestClassifier(random_state=42)
	rf_clf.fit(X_train, y_train)
	with open('rainier_model.pickle','wb') as f:
		pickle.dump(rf_clf,f)
	# print_score(rf_clf, X_train, y_train, cv=10)
	#
	# # print(X_test)
	# predictions = rf_clf.predict(X_test)
	# result = pd.DataFrame(X_test)
	# result["result"]=predictions
	# print(result)
	# predictions = rf_clf.predict([""])

main()
