import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dogs = pd.read_csv('train.csv')

dogs=dogs.drop(['issue_date', 'listing_date','pet_id'], axis=1)
lb_make = LabelEncoder()
dogs["color_type"] = lb_make.fit_transform(dogs["color_type"])
dogs['condition'].fillna(5,inplace=True)

x_train=dogs[['condition','color_type','length(m)','height(cm)','X1','X2']]
y_train=dogs[['breed_category']]

clf = RandomForestRegressor()
clf.fit(x_train, y_train)

x_train=dogs[['condition','color_type','length(m)','height(cm)','X1','X2','breed_category']]
y_train=dogs[['pet_category']]

clf2 = RandomForestRegressor()
clf2.fit(x_train,y_train)

dogs_test=pd.read_csv('test.csv')
Result=dogs_test[['pet_id']]

dogs_test=dogs_test.drop(['issue_date', 'listing_date','pet_id'], axis=1)
lb_make = LabelEncoder()
dogs_test["color_type"] = lb_make.fit_transform(dogs_test["color_type"])
dogs_test['condition'].fillna(5,inplace=True)

x_test=dogs_test[['condition','color_type','length(m)','height(cm)','X1','X2']]

RES=(clf.predict(x_test))
Result['breed_category']=RES
x_test['breed_category']=RES

Result['pet_category']=clf2.predict(x_test)

Result["breed_category"]=Result["breed_category"].astype(int)
Result["pet_category"]=Result["breed_category"].astype(int)

Result.to_csv('Result1.csv', header=True, index=False)
