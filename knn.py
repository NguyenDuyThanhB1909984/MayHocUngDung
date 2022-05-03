import pandas as pd 

#đọc dữ liệu 
dt = pd.read_csv("Spam.csv")
X = dt.iloc[:,1:10]
y = dt.spam

#đếm y = spam có bao nhiêu phần tử = 0, bao nhiêu = 1
print (dt['spam'].value_counts())


#nghi thức đánh giá hold out
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3.0, random_state = 10)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# K láng giềng,  vòng lặp để thay dổi k cho dễ
K = [5]
for k in K :

    Knn = KNeighborsClassifier(n_neighbors= k)
    Knn.fit(X_train,y_train)

    y_pred = Knn.predict(X_test)
 

   #tính độ chính xác tổng thể
    print ("Do 9 xac Knn" , round(accuracy_score(y_test,y_pred)*100,2),  "voi k = ",k)


    #tính độ chính xác cho từng phân lớp.
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
  
    df_confusion = pd.crosstab(y_test, y_pred,rownames=['Actual'], colnames=['Predicted'], margins=True)
    print (df_confusion)
    prec = tp /(tp +fp)
    rec =  tp /(tp +fn)
    f1 = (2*prec*rec)/(prec + rec)
    print ("Precision = " ,round(prec,2), " Recall = ",round(rec,2)," F1-score = " ,round(f1,2))
    print ("===============================")

## NATIVEBAYES
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred2 = model.predict(X_test)


#tính độ chính xác tổng thể
print ("Do 9 xac Native Bayes " , round(accuracy_score(y_test,y_pred2)*100,2))

#tính độ chính xác cho từng phân lớp.    
tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred2).ravel()

df_confusion = pd.crosstab(y_test, y_pred2,rownames=['Actual'], colnames=['Predicted'], margins=True)
print (df_confusion)
prec2 = tp2 /(tp2 +fp2)
rec2 =  tp2 /(tp2 +fn2)
f12 = (2*prec2*rec2)/(prec2 + rec2)
print ("Precision = " ,round(prec2,2), " Recall = ",round(rec2,2)," F1-score = " ,round(f12,2))
print ("===============================")



#Cay quyet dinh
from sklearn.tree import DecisionTreeClassifier
myTree = DecisionTreeClassifier(criterion= "gini", random_state= 100, max_depth = 3, min_samples_leaf= 5)
myTree.fit(X_train, y_train)
y_pred3 = myTree.predict(X_test)

#tính độ chính xác tổng thể
print ("Do 9 xac Decision Tree  " , round(accuracy_score(y_test,y_pred3)*100,2))

#tính độ chính xác cho từng phân lớp.    
tn3, fp3, fn3, tp3 = confusion_matrix(y_test, y_pred3).ravel()

df_confusion = pd.crosstab(y_test, y_pred3,rownames=['Actual'], colnames=['Predicted'], margins=True)
print (df_confusion)
prec3 = tp3 /(tp3 +fp3)
rec3 =  tp3 /(tp3 +fn3)
f13 = (2*prec3*rec3)/(prec3 + rec3)
print ("Precision = " ,round(prec3,2), " Recall = ",round(rec3,2)," F1-score = " ,round(f13,2))
print ("===============================")


