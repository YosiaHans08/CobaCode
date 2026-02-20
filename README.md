# ===============================
# 1. IMPORT LIBRARY
# ===============================
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===============================
# 2. LOAD DATASET
# ===============================
df = pd.read_csv("dataset.csv")   # ganti dengan file kamu

# ===============================
# 3. TEXT PREPROCESSING
# ===============================
def normalize(text):
    text = str(text).lower()

    # angka → huruf
    text = text.replace("0","o").replace("1","i").replace("3","e").replace("4","a")

    # hapus simbol
    text = re.sub(r'[^a-z\s]', '', text)

    # huruf berulang → max 2
    text = re.sub(r'(.)\1+', r'\1\1', text)

    return text

df["clean_text"] = df["text"].apply(normalize)

# ===============================
# 4. SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ===============================
# 5. VECTORIZER (CONTEXT AWARE)
# ===============================
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    analyzer="word",
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# 6. TRAIN MODEL
# ===============================
model = GradientBoostingClassifier()
model.fit(X_train_vec, y_train)

# ===============================
# 7. PREDICTION
# ===============================
y_pred = model.predict(X_test_vec)

# ===============================
# 8. EVALUATION METRICS
# ===============================
acc = accuracy_score(y_test, y_pred)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)

# ===============================
# 9. SAVE WRONG PREDICTIONS
# ===============================
errors = pd.DataFrame({
    "text": X_test,
    "actual": y_test,
    "predicted": y_pred
})

errors = errors[errors["actual"] != errors["predicted"]]

# ===============================
# 10. EXPORT REPORT TO EXCEL
# ===============================
with pd.ExcelWriter("model_report.xlsx") as writer:
    report_df.to_excel(writer, sheet_name="classification_report")
    cm_df.to_excel(writer, sheet_name="confusion_matrix")
    errors.to_excel(writer, sheet_name="wrong_predictions")

print("✅ Training selesai!")
print("Accuracy:", acc)
print("📊 Report tersimpan → model_report.xlsx")
