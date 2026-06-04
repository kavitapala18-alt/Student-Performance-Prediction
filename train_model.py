import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/student_data.csv")

X = df[["StudyHours", "Attendance", "PreviousScore"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

new_student = [[6, 88, 75]]
prediction = model.predict(new_student)

print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")