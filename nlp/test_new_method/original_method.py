import pandas as pd

df = pd.read_csv('nlp/test_new_method/Books_rating.csv', nrows=2000)

# 用評分對應情感標籤
def label_sentiment(score):
    if score >= 4:
        return 'positive'
    elif score <= 2:
        return 'negative'
    else:
        return 'neutral'

df = df[['review/text', 'review/score']].dropna()
df['sentiment'] = df['review/score'].apply(label_sentiment)
df.to_csv('nlp/test_new_method/labeled_reviews.csv', index=False)
print(df['sentiment'].value_counts())