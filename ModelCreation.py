import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('diabetes.csv')
df.head()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']],
                                                    df['Outcome'], test_size=0.3, random_state=109)

# Creating the model
logisticRegr = LogisticRegression(C=1)
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)

# Saving the Model
pickle_out = open("logisticRegr.pkl", "wb")
pickle.dump(logisticRegr, pickle_out)
pickle_out.close()
