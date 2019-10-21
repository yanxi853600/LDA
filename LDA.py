#(11/06 研究方法_HOMEWORK) 有用LDA 和 沒用LDA差別 (accuracy)
#LDA(類別)-監督式學習:有(x,y)投影到一個地方(x軸-新坐標系 低維)，做分類達到降維。
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
#載入數據集
avocado=pd.read_csv("avocado_2.csv")
x = pd.DataFrame(avocado,columns=["Total Volume","AveragePrice"])

#資料預處理
label_Encoder=preprocessing.LabelEncoder()
y=label_Encoder.fit_transform(avocado["type"])

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)

#特徵縮放 
sc = StandardScaler()  
train_x = sc.fit_transform(train_x)  
test_x = sc.transform(test_x)

#Accuracy of LDA
clf = LDA(n_components=1) #單線性判斷
fit_clf=clf.fit(train_x, train_y)
print('Accuracy of LDA classifier on training set: {:.3f}'.format(fit_clf.score(train_x,train_y)))
print('Accuracy of LDA classifier on test set: {:.3f}'.format(fit_clf.score(test_x,test_y)))

#用 LDA 降維後的特徵集來訓練隨機森林分類器
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(train_x, train_y)  
y_pred = classifier.predict(test_x)

#使用混淆矩陣評估性能
cm = confusion_matrix(test_y, y_pred)  
print('混淆矩陣:\n',cm)  
print('Accuracy : ' + str(accuracy_score(test_y, y_pred)))
