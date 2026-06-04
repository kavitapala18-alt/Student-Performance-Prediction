import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load CSV from repository root (one level above `src/`)
data_path = Path(__file__).resolve().parent.parent / "student_data.csv"
df = pd.read_csv(data_path)

X = df[["StudyHours", "Attendance", "PreviousScore"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))