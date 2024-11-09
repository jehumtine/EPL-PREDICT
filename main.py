import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

# Loading and preprocessing data
matches = pd.read_csv('matches.csv', index_col=0)
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["team_code"] = matches["team"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")


# Generate the mapping
#team_mapping = dict(enumerate(matches["opponent"].astype("category").cat.categories))
# Saving team_mapping to a JSON file
#with open("team_mapping.json", "w") as f:
#    json.dump(team_mapping, f)

# Loading team_mapping from the JSON file
with open("team_mapping.json", "r") as f:
    team_mapping = json.load(f)


# Store it for later use
matches["opp_code"] = matches["opponent"].map({v: k for k, v in team_mapping.items()})
matches["team_code"] = matches["team"].map({v: k for k, v in team_mapping.items()})
# Initializing the RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [1, 5, 10]
}
rf = RandomForestClassifier(n_estimators=200, min_samples_split=30, random_state=1,max_depth=5,min_samples_leaf=10)
# Spliting data into training and testing sets based on the date
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']

# List of predictor columns
predictors = ["venue_code", "opp_code", "hour", "day_code","xg","xga","team_code"]

# Training RandomForestClassifier on the training data
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])

combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])

precision_score(test["target"], preds)

# Function to compute rolling averages for specified columns
def rolling_averages(group, cols, new_cols):
  group = group.sort_values("date")
  rolling_stats = group[cols].rolling(3, closed='left').mean()
  group[new_cols] = rolling_stats
  group = group.dropna(subset=new_cols)
  return group

# Columns for which to calculate rolling averages
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]

# New column names for the rolling averages
new_cols = [f"{c}_rolling" for c in cols]

# Computing rolling averages for each team and merging results
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# Function to train the model and make predictions
def make_predictions(data, predictors):
  train = data[data["date"] < '2022-01-01']
  test = data[data["date"] > '2022-01-01']
  rf.fit(train[predictors], train["target"])
  preds = rf.predict(test[predictors])
  combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
  precision = precision_score(test["target"], preds)
  print(f'precision: {precision}')
  return combined, precision


def visualize_results(actual, predictions, filename='confusion_matrix.png'):
    """
    Visualizes the results of predictions using a confusion matrix and saves it as an image file.

    Parameters:
    - actual: Series or array-like, the actual target values.
    - predictions: Series or array-like, the predicted values by the model.
    - filename: str, name of the file to save the plot.
    """
    cm = confusion_matrix(actual, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Loss/Draw', 'Predicted Win'],
                yticklabels=['Actual Loss/Draw', 'Actual Win'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filename)  # Save the plot as a file
    plt.close()  # Close the plot to free up memory

# Call the function, which will save the plot as an image file
visualize_results(combined["actual"], combined["prediction"])


def plot_precision_recall(actual, predictions_prob):
    """
    Plots the precision-recall curve.

    Parameters:
    - actual: Series or array-like, the actual target values.
    - predictions_prob: array-like, predicted probabilities for the positive class.
    """
    precision, recall, _ = precision_recall_curve(actual, predictions_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()


# Making predictions using the updated dataset with rolling averages
combined, precision = make_predictions(matches_rolling, predictors + new_cols)
visualize_results(combined["actual"], combined["prediction"])
# Merging additional match details for better interpretation of results
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

# Class to handle missing values in dictionary mapping
class MissingDict(dict):
  __missing__ = lambda self, key: key

# Dictionary to map team names to shortened versions
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}

mapping = MissingDict(**map_values)

# Merging the predictions to compare scenarios where one team is predicted to win while the other is not
combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
merged[(merged["prediction_x"] == 1) & (merged["prediction_y"] == 0)]["actual_x"].value_counts()
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': predictors+new_cols,
    'importance': importances
}).sort_values(by='importance', ascending=False)
print(feature_importance_df)
# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_df['importance'], y=feature_importance_df['feature'])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
#grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='precision')
#grid_search.fit(train[predictors], train["target"])
#best_rf = grid_search.best_estimator_
#print(f'best rf = {best_rf}')


# Save the model to a file
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
