import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
#importing warnings library.
warnings.filterwarnings('ignore')
## Ignore warning
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.multiclass import unique_labels


print(os.listdir("./input/"))
print(os.listdir("./output"))
pd.set_option("display.width", None)
pd.set_option('display.max_rows', 100)

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
print(pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending=False)))
print(train.sample(5))
print(test.sample(5))
passengerid = test.PassengerId
print(train.info())
print("*" * 40)
print(test.info())


total = train.isnull().sum().sort_values(ascending=False)

percent = round(train.isnull().sum().sort_values(ascending=False)/len(train)*100, 2)
# 计算缺失率
print(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))
# concat链接表格数据

print("*" * 40)
percent = pd.DataFrame(round(train.Embarked.value_counts(dropna=False, normalize=True)*100, 2))
## creating a df with th
total = pd.DataFrame(train.Embarked.value_counts(dropna=False))
## concating percent and total dataframe

total.columns = ["Total"]
percent.columns = ['Percent']
print(pd.concat([total, percent], axis=1))
print("*" * 40)
print(train[train.Embarked.isnull()])
print("*" * 40)
plt.rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(figsize=(10, 8), ncols=2)
ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train, ax=ax[0])
ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test, ax=ax[1])
ax1.set_title("Training Set", fontsize=18)
ax2.set_title('Test Set',  fontsize=18)
ax1.axhline(y=80, ls="--", color="r", label="y=80")
ax1.text(1,80, s="y=80", color="b", fontdict={'size': 15, 'color':  'red'})
# fig.show()
print("*" * 40)
train.Embarked.fillna("C", inplace=True)
print("Fill the na of the Embarked")
print("groupby the data")

survivers = train.Survived

train.drop(["Survived"], axis=1, inplace=True)
all_data = pd.concat([train, test], ignore_index=False)

all_data.Cabin.fillna("N", inplace=True)
all_data.Cabin = [i[0] for i in all_data.Cabin]
with_N = all_data[all_data.Cabin == "N"]
without_N = all_data[all_data.Cabin != "N"]
print(all_data.groupby("Cabin")['Fare'].mean().sort_values())
# coding utf-8
# func cabin_estimator()


def cabin_estimator(i):
    a = 0
    if i < 16:
        a = "G"
    elif 27 > i >= 16:
        a = "F"
    elif 38 > i >= 27:
        a = "T"
    elif 47 > i >= 38:
        a = "A"
    elif 53 > i >= 47:
        a = "E"
    elif 54 > i >= 53:
        a = "D"
    elif 116 > i >= 54:
        a = "C"
    else:
        a = "B"
    return a


# applying cabin estimator function.
with_N['Cabin'] = with_N.Fare.apply(lambda x: cabin_estimator(x))

all_data = pd.concat([with_N, without_N], axis=0)

all_data.sort_values(by='PassengerId', inplace=True)
train = all_data[:891]
test = all_data[891:]

train['Survived'] = survivers
print(test[test.Fare.isnull()])

missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()
test.Fare.fillna(missing_value, inplace=True)

print("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
print("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))


pal = {'male': "Green", 'female': "Pink"}
plt.subplots(figsize=(10, 8))
print(train.sample(5))
ax = sns.barplot(x="Sex",
                 y="Survived",
                 data=train,
                 palette=pal,
                 linewidth=2)
plt.title("Survived/Non-Survived Gender distributon", fontsize=25)
plt.ylabel("% of survived", fontsize=15)
plt.xlabel("Sex", fontsize=15)
#plt.show()

pl = {1: "seagreen", 0: "gray"}
sns.set(style="darkgrid")
plt.subplots(figsize=(10, 8))
ax = sns.countplot(x="Sex", hue="Survived", data=train,
                   linewidth=2,
                   palette=pl)
plt.title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 25)
plt.xlabel("Sex", fontsize=15);
plt.ylabel("# of Passenger Survived", fontsize=15)
leg = ax.get_legend()
leg.set_title("Survived")
legs = leg.texts
legs[0].set_text("No")
legs[1].set_text("Yes")
# plt.show()

# barplot of Pclass
plt.subplots(figsize = (10,8))
sns.barplot(x = "Pclass",
            y = "Survived",
            data=train,
            linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper', 'Middle', 'Lower']
val = [0, 1, 2]  # this is just a temporary trick to get the label right.
plt.xticks(val, labels)
# plt.show()

# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
# I have included to different ways to code a plot below, choose the one that suites you.
ax=sns.kdeplot(train.Pclass[train.Survived == 0] ,
               color='gray',
               shade=True,
               label='not survived')
ax = sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] ,
               color='g',
               shade=True,
               label='survived')
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Class", fontsize = 15)
# Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(train.Pclass.unique()), labels)

# plt.show()

# Kernel Density Plot
fig = plt.figure(figsize=(15, 8),)
ax = sns.kdeplot(train.loc[(train['Survived'] == 0), 'Fare'], color='gray', shade=True, label='not survived')
ax = sns.kdeplot(train.loc[(train['Survived'] == 1), 'Fare'], color='g', shade=True, label='survived')
plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Fare", fontsize=15)
# fig.show()
print(train[train.Fare > 280])

# Kernel Density Plot
fig = plt.figure(figsize=(8,5),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 25)
plt.xlabel("Age", fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
# fig.show()


pal = {1:"seagreen", 0:"gray"}
g = sns.FacetGrid(train,size=4, col="Sex", row="Survived", margin_titles=True, hue="Survived",
                  palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white')
g.fig.suptitle("Survived by Sex and Age", size = 15)
plt.subplots_adjust(top=0.90)
# plt.show()


g = sns.FacetGrid(train,size=4, col="Sex", row="Embarked", margin_titles=True, hue="Survived",
                  palette = pal
                  )
g = g.map(plt.hist, "Age", edgecolor='white').add_legend()
g.fig.suptitle("Survived by Sex and Age", size=15)
plt.subplots_adjust(top=2,right=50)
# plt.show()

g = sns.FacetGrid(train, size=4,hue="Survived", col ="Sex", margin_titles=True,
                palette=pal,)
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
g.fig.suptitle("Survived by Sex, Fare and Age", size = 10)
plt.subplots_adjust(top=0.85)
# plt.show()
#  dropping the three outliers where Fare is over $500
train = train[train.Fare < 500]
# factor plot

plt.title("Factorplot of Parents/Children survived", fontsize = 25)
plt.subplots_adjust(top=0.85)

sns.factorplot(x =  "SibSp", y = "Survived", data = train,kind = "point",size = 8)
plt.title('Factorplot of Sibilings/Spouses survived', fontsize = 25)
plt.subplots_adjust(top=0.85)


print(train.describe())
print(train.describe(include =['O']))
train.loc[train['Sex'] == 'male', 'Sex'] = 1
train.loc[train['Sex'] == 'female', 'Sex'] = 0
test.loc[test['Sex'] == 'male', 'Sex'] = 1
test.loc[test['Sex'] == 'female', 'Sex'] = 0
# 二值化处理性别，pandas的corr计算只会统计数值类型,将测试集合和训练集合均修改
print(train.sample(5))
print(pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending=False)))
print("*"*40)
corr = train.corr()**2
print(corr.Survived.sort_values(ascending=False))

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(train.corr(), dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize=(10,8))
sns.heatmap(train.corr(),
            annot=True,
            cmap = 'RdBu_r',
            linewidths=0.1,
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y=1.03,fontsize=20)
# plt.show()

# feature engineering
# Creating a new colomn with a
train['name_length'] = [len(i) for i in train.Name]
test['name_length'] = [len(i) for i in test.Name]


def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a


train['nLength_group'] = train['name_length'].map(name_length_group)
test['nLength_group'] = test['name_length'].map(name_length_group)

print(train.sample(1))
print(test.sample(1))
# name length over

# get the title from the name
train["title"] = [i.split('.')[0] for i in train.Name]
train["title"] = [i.split(',')[1] for i in train.title]
test["title"] = [i.split('.')[0] for i in test.Name]
test["title"]= [i.split(',')[1] for i in test.title]

# train Data
train["title"] = [i.replace('Ms', 'Miss') for i in train.title]
train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]
train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]
train["title"] = [i.replace('Dr', 'rare') for i in train.title]
train["title"] = [i.replace('Col', 'rare') for i in train.title]
train["title"] = [i.replace('Major', 'rare') for i in train.title]
train["title"] = [i.replace('Don', 'rare') for i in train.title]
train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]
train["title"] = [i.replace('Sir', 'rare') for i in train.title]
train["title"] = [i.replace('Lady', 'rare') for i in train.title]
train["title"] = [i.replace('Capt', 'rare') for i in train.title]
train["title"] = [i.replace('the Countess', 'rare') for i in train.title]
train["title"] = [i.replace('Rev', 'rare') for i in train.title]


# test data
test['title'] = [i.replace('Ms', 'Miss') for i in test.title]
test['title'] = [i.replace('Dr', 'rare') for i in test.title]
test['title'] = [i.replace('Col', 'rare') for i in test.title]
test['title'] = [i.replace('Dona', 'rare') for i in test.title]
test['title'] = [i.replace('Rev', 'rare') for i in test.title]
# title feature over

## Family_size seems like a good feature to create
train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1


def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a


train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)

train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]

print(train.Ticket.value_counts().sample(10))
# drop the ticket feature
train.drop(['Ticket'], axis=1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)

# fare feature
# Calculating fare based on family size.
train['calculated_fare'] = train.Fare/train.family_size
test['calculated_fare'] = test.Fare/test.family_size


def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a


train['fare_group'] = train['calculated_fare'].map(fare_group)
test['fare_group'] = test['calculated_fare'].map(fare_group)
# fare feature

train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)


train = pd.get_dummies(train, columns=['title',"Pclass", 'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)
test = pd.get_dummies(test, columns=['title',"Pclass",'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)
train.drop(['family_size','Name', 'Fare','name_length'], axis=1, inplace=True)
test.drop(['Name','family_size',"Fare",'name_length'], axis=1, inplace=True)


# age group
# rearranging the columns so that I can easily use the dataframe to predict the missing age values.
train = pd.concat([train[["Survived", "Age", "Sex","SibSp","Parch"]], train.loc[:,"is_alone":]], axis=1)
test = pd.concat([test[["Age", "Sex"]], test.loc[:,"SibSp":]], axis=1)
# print(train.sample(5))
# print(test.sample(5))


# writing a function that takes a dataframe with missing values and outputs it by filling the missing values.
def completing_age(df):
    # getting all the features except survived
    age_df = df.loc[:, "Age":]

    temp_train = age_df.loc[age_df.Age.notnull()]  ## df with age values
    temp_test = age_df.loc[age_df.Age.isnull()]  ## df without age values

    y = temp_train.Age.values  ## setting target variables(age) in y
    x = temp_train.loc[:, "Sex":].values

    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
    rfr.fit(x, y)

    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])

    df.loc[df.Age.isnull(), "Age"] = predicted_age

    return df


# Implementing the completing_age function in both train and test dataset.
completing_age(train)
completing_age(test)

# print(train.sample(5))
# print(test.sample(5))
# Let's look at the his
plt.subplots(figsize = (22,10),)
sns.distplot(train.Age, bins = 100, kde = True, rug = False, norm_hist=False)
# plt.show()

# print(train[train.Age.isnull()])
# The output is Empty dataframe


# create bins for age
def age_group_fun(age):
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 4:
        a = 'toddler'
    elif age <= 13:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 35:
        a = 'Young_Adult'
    elif age <= 45:
        a = 'adult'
    elif age <= 55:
        a = 'middle_aged'
    elif age <= 65:
        a = 'senior_citizen'
    else:
        a = 'old'
    return a


# Applying "age_group_fun" function to the "Age" column.
train['age_group'] = train['Age'].map(age_group_fun)
test['age_group'] = test['Age'].map(age_group_fun)

# Creating dummies for "age_group" feature.
train = pd.get_dummies(train, columns=['age_group'], drop_first=True)
test = pd.get_dummies(test, columns=['age_group'], drop_first=True)

# train.drop('Age', axis=1, inplace=True)
# test.drop('Age', axis=1, inplace=True)

# pre modeling
# separating our independent and dependent variable
X = train.drop(['Survived'], axis=1)
y = train["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state=0)

headers = X_train.columns

print(X_train.head())
# Feature Scaling
## We will be using standardscaler to transform
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# transforming "train_x"
X_train = sc.fit_transform(X_train)
# transforming "test_x"
X_test = sc.transform(X_test)
# transforming "The testset"
test = sc.transform(test)

print(pd.DataFrame(X_train, columns=headers).head())


# modeling the data

train.calculated_fare = train.calculated_fare.astype(float)
plt.subplots(figsize = (12,10))
plt.scatter(train.Age, train.Survived);
plt.xlabel("Age")
plt.ylabel('Survival Status')
# plt.show()


# import LogisticRegression model in python.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score

## call on the model object
logreg = LogisticRegression(solver='liblinear')

## fit the model with "train_x" and "train_y"
logreg.fit(X_train,y_train)

## Once the model is trained we want to find out how well the model is performing, so we test the model.
## we use "test_x" portion of the data(this data was not used to fit the model) to predict model outcome.
y_pred = logreg.predict(X_test)

## Once predicted we save that outcome in "y_pred" variable.
## Then we compare the predicted value( "y_pred") and actual value("test_y") to see how well our model is performing.


print("Score is: {}".format(round(accuracy_score(y_pred, y_test), 4)))



from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

[unique_labels(y_test, y_pred)]


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

class_names = np.array(['not_survived', 'survived'])

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

# plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# ROC
from sklearn.metrics import roc_curve, auc
#plt.style.use('seaborn-pastel')
y_score = logreg.decision_function(X_test)

FPR, TPR, _ = roc_curve(y_test, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Titanic survivors', fontsize= 18)
# plt.show()
# recall
from sklearn.metrics import precision_recall_curve

y_score = logreg.decision_function(X_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)
plt.legend(loc="lower right")
# plt.show()

# Using StratifiedShuffleSplit
# We can use KFold, StratifiedShuffleSplit, StratiriedKFold or ShuffleSplit, They are all close cousins. look at sklearn userguide for more info.
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
# Using standard scale for the whole dataset.

# saving the feature names for decision tree display
column_names = X.columns

X = sc.fit_transform(X)
accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X,y, cv  = cv)
print ("Cross-Validation accuracy scores:{}".format(accuracies))
print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))


from sklearn.model_selection import GridSearchCV, StratifiedKFold

# if __name__=='__main__':

# C_vals is the alpla value of lasso and ridge regression(as alpha increases the model complexity decreases,)
# remember effective alpha scores are 0<alpha<infinity
C_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 16.5, 17, 17.5, 18]
# Choosing penalties(Lasso(l1) or Ridge(l2))
penalties = ['l1', 'l2']
# Choose a cross validation strategy.
cv = StratifiedShuffleSplit(n_splits=10, test_size=.25)

# setting param for param_grid in GridSearchCV.
param = {'penalty': penalties, 'C': C_vals}

logreg = LogisticRegression(solver='liblinear')
# Calling on GridSearchCV object.
grid = GridSearchCV(estimator=LogisticRegression(),
                    param_grid=param,
                    scoring='accuracy',
                    n_jobs=1,
                    cv=cv
                    )
# Fitting the model
grid.fit(X, y)

## Getting the best of everything.
    # print(grid.best_score_)
    # print(grid.best_params_)
    # print(grid.best_estimator_)

### Using the best parameters from the grid-search.
logreg_grid = grid.best_estimator_
    # print(logreg_grid.score(X, y))


test_pre = logreg_grid.predict(test)
submission = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": test_pre })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic1_submission.csv", index=False)
