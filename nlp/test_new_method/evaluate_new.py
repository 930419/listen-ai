import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 載入資料
df = pd.read_csv(r'C:\Users\jolin\GitHub\listen-ai\nlp\test_new_method\labeled_reviews.csv')

X = df['review/text'].astype(str)
y = df['sentiment']

# 切分訓練集和測試集（80% 訓練，20% 測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF 向量化
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 訓練模型
start = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
train_time = time.time() - start

# 預測
start = time.time()
y_pred = model.predict(X_test_tfidf)
predict_time = time.time() - start

# 評估
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score (weighted):", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\n訓練時間: {train_time:.2f} 秒")
print(f"預測時間 (400筆): {predict_time:.4f} 秒")