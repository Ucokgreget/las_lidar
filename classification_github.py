import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, RidgeCV, Lasso
from sklearn.metrics import accuracy_score


dt=pd.read_csv("290290.txt")
dt.Classification.value_counts()
dt.describe()

"""for each lidar point:
 neighborhood = points closer than d (you could cut it down to a box by saying x' < x +/- d; y' < y +/- d; etc. 
 then just use a little Pythagorus to do the rest) 
 for each neighborhood point:
  calculate covariance
 add up covariance matrix
 classify based on the eigenvalue
 """
"""
big_array = [] #  empty regular list
for i in range(5):
    arr = i*np.ones((2,4)) # for instance
    big_array.append(arr)
big_np_array = np.array(big_array) 
"""



#D is the distance parameter. Use this to determine neighborhood proximity for the points in the cloud

D=2
neighborhood=np.array()

for point in range(0,13402173): 
    X=dt['X'][point]
    Y=dt['Y'][point]
    Z=dt['Z'][point]
    for i in range (0, 13402173):
        X_i=dt['X'][i]
        Y_i=dt['Y'][i]
        Z_i=dt['Z'][i]
        if X < X_i - D and X > X_i + D:
            if Y < Y_i - D and Y > Y_i + D:
                if Z < Z_i - D and Z > Z_i + D:
                    neighborhood.append([X_i, Y_i, Z_i])


x_train=[]
y_train=[] 
x_test=[]
y_actual=[]
x1_train=[]
y1_train=[] 

for i in range(0,13402173): 
    x1_train.append([dt['X'][i], dt['Y'][i], dt['Z'][i], dt['Intensity'][i], dt['ReturnNumber'][i], dt['NumberOfReturns'][i], dt['ScanDirectionFlag'][i], dt['EdgeOfFlightLine'][i], dt['ScanAngleRank'][i], dt['UserData'][i], dt['PointSourceId'][i], dt['Time'][i]])
    y1_train.append(dt['Classification'][i])
    if dt["Classification"][i]!= 1 and dt["Classification"][i]!= 12:
        x_train.append([dt['X'][i], dt['Y'][i], dt['Z'][i], dt['Intensity'][i], dt['ReturnNumber'][i], dt['NumberOfReturns'][i], dt['ScanDirectionFlag'][i], dt['EdgeOfFlightLine'][i], dt['ScanAngleRank'][i], dt['UserData'][i], dt['PointSourceId'][i], dt['Time'][i]])
        y_train.append(dt['Classification'][i])
        
    if dt["Classification"][i]==12 or dt["Classification"][i]==1:
        x_test.append([dt['X'][i], dt['Y'][i], dt['Z'][i], dt['Intensity'][i], dt['ReturnNumber'][i], dt['NumberOfReturns'][i], dt['ScanDirectionFlag'][i], dt['EdgeOfFlightLine'][i], dt['ScanAngleRank'][i], dt['UserData'][i], dt['PointSourceId'][i], dt['Time'][i]])
        y_actual.append(dt['Classification'][i])



print(len(x_train), len(y_train))
print(len(x_test), len(y_actual))


X_train, X_test,Y_train, Y_test= cross_validation.train_test_split(x_train,y_train,test_size=0.7)
                    
# I=[0.001,0.005,0.01, 0.05, 0.1, 0.15,0.4,0.5,0.6,0.7,0.9,1]

# clfs=[KNeighborsClassifier(2),
#       KNeighborsClassifier(9),
#       AdaBoostClassifier()]

# # GDClassifier(loss="hinge", penalty ="l1"),
# # DecisionTreeClassifier(max_depth=3),
# # svm.SVC(kernel="linear", C=0.5),
# # svm.SVC(kernel="rbf", C=0.5), 
# # BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=True),
# #  RandomForestClassifier(max_depth=4, n_estimators=10, max_features=1),

# for clf in clfs:
#     clf.fit(X_train, Y_train)
#     Y_pred=clf.predict(X_test)
#     accuracy=accuracy_score(Y_test, Y_pred)
#     print clf,accuracy



clf = KNeighborsClassifier(9)
clf.fit(x_train, y_train)
Y_pred=clf.predict(x_test)
accuracy_score(y_actual, Y_pred)


len(Y_pred)


x = x_train+x_test
len(x)


print (type(y_train))
print (type(Y_pred))
print (len(y_train))
print (len(Y_pred))
y = y_train + Y_pred.tolist()
print (len(y))


import time
start_time=time.time()

# final = []
# for i in range(len(x_train)):
#     final.append([x_train[i][0],x_train[i][1], x_train[i][2], x_train[i][3], x_train[i][4], x_train[i][5], x_train[i][6], x_train[i][7], y_train[i], x_train[i][8], x_train[i][9], x_train[i][10], x_train[i][11]])   
# for i in range(len(x_test)):
#     final.append([x_test[i][0],x_test[i][1], x_test[i][2], x_test[i][3], x_test[i][4], x_test[i][5], x_test[i][6], x_test[i][7], Y_pred.tolist()[i], x_test[i][8], x_test[i][9], x_test[i][10], x_test[i][11]])
#     if i%1000==0:
#         print str(i*100.0/len(x_test))+"% complete"
# print len(final)
# end_time=time.time()-start_time
# print end_time

h = ['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns', 'ScanDirectionFlag', 'EdgeOfFlightLine', 'ScanAngleRank', 'UserData', 'PointSourceId', 'Time']
final_train = pd.DataFrame(x_train, columns = h)
final_train.insert(8, "Classification", y_train)

len(final_train)


final_test = pd.DataFrame(x_test, columns = h)
final_test.insert(8, "Classification", Y_pred)
len(final_test)


frames = [final_train, final_test]
result = pd.concat(frames)

# h = ['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns', 'ScanDirectionFlag', 'EdgeOfFlightLine', 'Classification', 'ScanAngleRank', 'UserData', 'PointSourceId', 'Time']

# pd.DataFrame(final, columns = h).
result.to_csv("290290_classified.txt", index=None, sep=",")



# pd.DataFrame(Y_pred).to_csv("a.csv")
# dt.head(2)



pd.DataFrame(Y_pred)[0].value_counts()




# x1_train=[]
# y1_train=[] 

# for i in range(0,800000): 
#     x1_train.append([dt['X'][i], dt['Y'][i], dt['Z'][i], dt['Intensity'][i], dt['ReturnNumber'][i], dt['NumberOfReturns'][i], dt['ScanDirectionFlag'][i], dt['EdgeOfFlightLine'][i], dt['ScanAngleRank'][i], dt['UserData'][i], dt['PointSourceId'][i], dt['Time'][i]])
#     y1_train.append(dt['Classification'][i])



# X_train, X_test,Y_train, Y_test= cross_validation.train_test_split(x1_train,y1_train,test_size=0.7)


# # In[41]:

# clf = KNeighborsClassifier(9)
# clf.fit(X_train, Y_train)
# Y_pred=clf.predict(X_test)
# accuracy_score(Y_test, Y_pred)


# # In[42]:

# pd.DataFrame(Y_pred)[0].value_counts()


# # In[43]:

# pd.DataFrame(Y_test)[0].value_counts()


# # In[ ]:



