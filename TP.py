import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Load the dataset
dataset = pd.read_csv("game_teams_stats.csv")

# Split the data into team and opposition
dataset_team = dataset[dataset['team_id'] == 10]
dataset_opp = dataset[dataset['team_id'] != 10]

# Merge team and opposition to get the team and the opponents they've played against
games = pd.merge(dataset_team, dataset_opp, on='game_id', how='left')
games = pd.DataFrame(games, columns=['game_id', 'HoA_x', 'won_x', 'goals_x', 'shots_x',
                                     'pim_x', 'powerPlayOpportunities_x', 'powerPlayGoals_x', 'faceOffWinPercentage_x', 'giveaways_x',
                                     'team_id_y', 'goals_y', 'shots_y', 'pim_y', 'powerPlayOpportunities_y', 'powerPlayGoals_y', 'faceOffWinPercentage_y', 'giveaways_y' ])

# Add features to help prediction
games["HoA_x"] = preprocessing.LabelEncoder().fit_transform(games["HoA_x"])
games["won_x"] = (games["won_x"].astype(int))
games["season"] = (games['game_id'].apply(str).str.slice(stop=4)).apply(int)
games["shot_ratio"] = (games['shots_x'] / (games['shots_x'] + games['shots_y']))
games["goal_ratio"] = (games['goals_x'] / (games['goals_x'] + games['goals_y']))
games["save_percentage"] = 1 - (games['goals_y'] / games['shots_y'])
games = games.fillna(value=0) # gets rid of division by zero in goal ratio
games.info() # checking for null values
print()

# plot each feature
info = games.drop(labels=['game_id', 'won_x', 'HoA_x', 'season'], axis=1)
fig, axs = plt.subplots(ncols=6, nrows=3, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k, v in info.items():
    sns.distplot(v, ax=axs[index])
    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
correlation_matrix = info.corr().round(2)
plt.show()
sns.heatmap(data=correlation_matrix, annot=True, )
plt.show()

# box plot shots
pd.DataFrame(games, columns=['shots_x', 'shots_y']).boxplot()
plt.xlabel('Team')
plt.ylabel('Shots')
plt.title('# of shots for and against')
plt.show()

# goals per season
goals_x = pd.DataFrame(games, columns=['season', 'goals_x'])
goals_y = pd.DataFrame(games, columns=['season', 'goals_y'])
goals_y.groupby(['season']).sum().plot(kind='bar')
goals_x.groupby(['season']).sum().plot(kind='bar')
plt.show()

# get training features and training data for 2010-2017 season
features = [feature for feature in games.columns.values if feature not in ['won_x', 'game_id', 'season']]
x = pd.DataFrame(games, columns=features)
y = pd.DataFrame(games, columns=['won_x'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.3)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('GNB', GaussianNB()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier(n_estimators=10)))
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=15)
    acc = cross_val_score(model, x_train, np.ravel(y_train), cv=kfold, scoring="accuracy")
    results.append(acc)
    names.append(name)
    print("%s: %0.5f (+/- %0.5f)" % (name, acc.mean(), acc.std()))

print()
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

lr = LogisticRegression(solver=)

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(x_train, np.ravel(y_train))
pred = lr.predict(x_test)
print("Accuracy: %0.5f" % (accuracy_score(y_test, pred)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

dtc = DecisionTreeClassifier()
dtc.fit(x_train, np.ravel(y_train))
pred = dtc.predict(x_test)
print("Accuracy: %0.5f" % (accuracy_score(y_test, pred)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(x_train, np.ravel(y_train))
pred = rfc.predict(x_test)
print("Accuracy: %0.5f" % (accuracy_score(y_test, pred)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

