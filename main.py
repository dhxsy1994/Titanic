import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
#importing warnings library.
warnings.filterwarnings('ignore')
## Ignore warning
import os

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
sns.factorplot(x = "Parch", y = "Survived", data = train,kind = "point",size = 8)
plt.title("Factorplot of Parents/Children survived", fontsize = 25)
plt.subplots_adjust(top=0.85)

sns.factorplot(x =  "SibSp", y = "Survived", data = train,kind = "point",size = 8)
plt.title('Factorplot of Sibilings/Spouses survived', fontsize = 25)
plt.subplots_adjust(top=0.85)


print(train.describe())
print(train.describe(include =['O']))
train.loc[train['Sex'] == 'male', 'Sex'] = 1
train.loc[train['Sex'] == 'female', 'Sex'] = 0
print(train.sample(5))
print(pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending=False)))
print("*"*40)
corr = train.corr()**2
print(corr.Survived.sort_values(ascending=False))

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(train.corr(), dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize=(10,8))
sns.heatmap(train.corr(),
            annot=True,
            cmap = 'RdBu_r',
            linewidths=0.1,
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y=1.03,fontsize=20)
plt.show()

