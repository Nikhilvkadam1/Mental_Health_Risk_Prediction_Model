import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("C:/Users/nikhi/OneDrive/Desktop/MY STUDY/SEMESTER 6/Classrom/ML/mental_health_condition_dataset.csv")
df.head(10)

X = df[['Age', 'Sleep Hours', 'Work Hours', 'Stress Level', 'Mood Rating']]
y = df['Risk Level']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, ccp_alpha=0.05, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier()
}

model_accuracies = []

for name, model in models.items():
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
    cv_accuracy = cv_scores.mean() * 100

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    model_accuracies.append({
        "Model": name,
        "Cross-Validation Accuracy (%)": cv_accuracy,
        "Test Set Accuracy (%)": accuracy
    })

accuracy_df = pd.DataFrame(model_accuracies)

accuracy_df.index = accuracy_df.index + 1

print("\nModel Accuracy Comparison:")
print(accuracy_df)

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Test Set Accuracy (%)", data=accuracy_df, hue="Model", legend=False)
plt.title("Model Accuracy Comparison")
plt.ylabel("Test Set Accuracy (%)")
plt.xticks(rotation=45)
plt.show()

best_model = DecisionTreeClassifier(max_depth=3, ccp_alpha=0.05, random_state=42)
best_model.fit(X_train, y_train)

print("\n🔍 Predict Your Mental Health Risk Level:")

try:
    age = int(input("Enter your age: "))
    if age < 5 or age > 100:
        raise ValueError("Age must be between 5 and 100.")

    sleep_hours = round(float(input("Enter average sleep hours: ")))
    work_hours = round(float(input("Enter average work hours: ")))
    stress = round(float(input("Enter stress level (1-10): ")))
    mood = round(float(input("Enter mood rating (1-10): ")))

    user_input_df = pd.DataFrame([{
        'Age': age,
        'Sleep Hours': sleep_hours,
        'Work Hours': work_hours,
        'Stress Level': stress,
        'Mood Rating': mood
    }])

    user_input_scaled = scaler.transform(user_input_df)

    prediction = best_model.predict(user_input_scaled)[0]
    predicted_label = le.inverse_transform([prediction])[0]

    print(f"\n🧠 Predicted Mental Health Risk Level: {predicted_label}")

except ValueError as e:
    print(f"❌ Invalid input: {e}")

risk_groups = df.groupby('Risk Level')[['Age', 'Sleep Hours', 'Work Hours', 'Stress Level', 'Mood Rating']].mean().round(2)

print("Average values for each Risk Level:\n")
print(risk_groups)