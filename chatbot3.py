import pandas as pd
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------- Step 1: Load Dataset from CSV -------------
CSV_FILE = 'intents.csv'  # Ensure this file exists

try:
    df = pd.read_csv(CSV_FILE)
    assert 'text' in df.columns and 'label' in df.columns
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# ----------- Step 2: Encode Labels ----------------------
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

# ----------- Step 3: Smart Train-Test Split -------------
num_classes = len(le.classes_)
min_test_size = max(num_classes, 5)  # At least 1 sample per class, or min 5
total_samples = len(df)

if total_samples < min_test_size * 2:
    warnings.warn("Dataset is too small for stratification. Using random split.")
    stratify = None
    test_size = 0.33  # Use 33% test data if small
else:
    stratify = df['label_enc']
    test_size = 0.2   # Standard 80/20 split

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_enc'], test_size=test_size, random_state=42, stratify=stratify)

# ----------- Step 4: Create Model Pipeline --------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# ----------- Step 5: Evaluate ---------------------------
y_pred = pipeline.predict(X_test)

all_labels = list(range(len(le.classes_)))

print("\n📊 Classification Report:\n", classification_report(
    y_test, y_pred, labels=all_labels, target_names=le.classes_, zero_division=0))

print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=all_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------- Step 6: Save Model -------------------------
joblib.dump(pipeline, 'chatbot_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# ----------- Step 7: Define Chatbot Responses ----------
responses = {
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Goodbye! Have a great day!",
    "weather": "It looks sunny right now. Want a detailed forecast?",
    "music": "Playing your favorite music.",
    "identity": "I'm a chatbot powered by machine learning!"
}

# ----------- Step 8: Chatbot Logic ---------------------
def get_response(user_input):
    try:
        label = pipeline.predict([user_input])[0]
        intent = le.inverse_transform([label])[0]
        return responses.get(intent, "Hmm... I didn't get that. Can you try again?")
    except Exception as e:
        return "Error processing input."

# ----------- Step 9: GUI with Tkinter ------------------
def send():
    user_input = entry.get()
    if user_input.strip() == "":
        return
    chat_log.insert(tk.END, "You: " + user_input + "\n")
    response = get_response(user_input)
    chat_log.insert(tk.END, "Bot: " + response + "\n\n")
    entry.delete(0, tk.END)

# GUI setup
root = tk.Tk()
root.title("Chatbot using Logistic Regression")

chat_log = tk.Text(root, bg="white", height=20, width=60, font=("Arial", 12))
chat_log.pack(padx=10, pady=10)

entry = tk.Entry(root, width=60, font=("Arial", 12))
entry.pack(padx=10, pady=(0, 10))

send_button = tk.Button(root, text="Send", width=12, font=("Arial", 12), command=send)
send_button.pack(pady=(0, 10))

root.mainloop()
