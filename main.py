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


print(os.listdir("./input/"))
print(os.listdir("./output"))
pd.set_option("display.width", None)
pd.set_option('display.max_rows', 100)

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
print(pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending=False)))
print(train.sample(5))
print(test.sample(5))

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

print(train.sample(5))
print(test.sample(5))
# Let's look at the his
plt.subplots(figsize = (22,10),)
sns.distplot(train.Age, bins = 100, kde = True, rug = False, norm_hist=False)
plt.show()
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

train.drop('Age', axis=1, inplace=True)
test.drop('Age', axis=1, inplace=True)


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

