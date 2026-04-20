import pandas as pd
import re
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 從 app.py 複製過來的字典和函式
POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "awesome", "happy", "amazing",
    "nice", "best", "positive", "fast", "smooth", "reliable",
}
POSITIVE_WORDS_ZH_TW = {
    "好", "很好", "優秀", "喜歡", "讚", "開心", "高興", "棒", "最佳", "正面",
    "快速", "順暢", "可靠", "滿意", "推薦",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "slow", "bug", "bugs",
    "issue", "issues", "angry", "broken", "negative", "expensive",
}
NEGATIVE_WORDS_ZH_TW = {
    "差", "糟糕", "很糟", "討厭", "最差", "慢", "錯誤", "問題", "生氣",
    "壞掉", "負面", "昂貴", "失望", "卡頓",
}
NEGATION_WORDS = {"not", "never", "no", "hardly", "不", "沒", "無", "未", "別", "不是"}

POSITIVE_WORDS_ALL = POSITIVE_WORDS | POSITIVE_WORDS_ZH_TW
NEGATIVE_WORDS_ALL = NEGATIVE_WORDS | NEGATIVE_WORDS_ZH_TW

def classify_text(text: str):
    tokens = re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]+", text.lower())
    score = 0
    previous_tokens = ["", ""]
    for token in tokens:
        is_negated = any(prev in NEGATION_WORDS for prev in previous_tokens)
        if token in POSITIVE_WORDS_ALL:
            score += -1 if is_negated else 1
        elif token in NEGATIVE_WORDS_ALL:
            score += 1 if is_negated else -1
        previous_tokens = [previous_tokens[-1], token]
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"

# 載入資料
df = pd.read_csv(r'C:\Users\jolin\GitHub\listen-ai\nlp\test_new_method\labeled_reviews.csv')

# 預測
df['predicted'] = df['review/text'].astype(str).apply(classify_text)

# 評估
print("Accuracy:", accuracy_score(df['sentiment'], df['predicted']))
print("F1-score (weighted):", f1_score(df['sentiment'], df['predicted'], average='weighted'))
print("\nClassification Report:")
print(classification_report(df['sentiment'], df['predicted']))