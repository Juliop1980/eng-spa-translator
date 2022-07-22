import pandas as pd
from sklearn.model_selection import train_test_split


# read csv input data
data = pd.read_csv('mixedtranslation.csv')
english_sentences = data['english_sentences']
spanish_sentences = data['spanish_sentences']
#print(data)
#train, test = train_test_split(df, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(english_sentences, spanish_sentences, test_size=0.2, random_state=1)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

print(X_train)
print("------------")
print(y_train)
""" # Drop first column of dataframe
df = df.iloc[:, 1:]
dataTypeDict = dict(df.dtypes)

df["spam"] = df["spam"].astype('category').cat.codes

train, test = train_test_split(df, test_size=0.2)

emails_list = df['text'].tolist()

vectorizer = CountVectorizer(min_df=0, lowercase=False)

vectorizer.fit(emails_list)
emails_list = df['text'].values
y = df['spam'].values

emails_train, emails_test, y_train, y_test = train_test_split(emails_list, y, test_size=0.2, random_state=1000) """