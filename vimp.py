import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
print('Libraries loaded')

# Load messages database
# drop --> id , created_at , user_id , color_id , uuid
# Remove punctuations
# lower case all the messages
messages = pd.read_csv('messages.csv')
print('Messages database loaded')

messages.drop(['id', 'created_at', 'user_id', 'color_id', 'uuid'], axis=1, inplace=True)
print('Dropped unneccessary columns')

# Function to remove punctuations
def remove_punct(text):
    punctuation = '''!()-[]{};:'"\\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in text:
        if char not in punctuation:
            no_punct = no_punct + char
    return no_punct

# Apply punctuation removal function to the original messages column
messages['message_text'] = messages['message_text'].apply(remove_punct)
print('Punctuations removed')

messages['message_text'] = messages['message_text'].str.lower()
print('Messages turned lowercase')

# Compute TF-IDF Vector
# Define TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
print('TF-IDF Vectorizer defined')

# Construct TF-IDF matrix by fitting and transforming data
tfidf_matrix = tfidf.fit_transform(messages['message_text'])
print('TF-IDF matrix created')

# Get the terms out
terms = tfidf.get_feature_names_out()

# Convert sparse tdidf matrix to dense matrix just for visualisation
dense_tfidf = tfidf_matrix.todense()

# Create a DataFrame with terms as columns and documents as rows
df_tfidf = pd.DataFrame(dense_tfidf, columns=terms)

# Sum the TF-IDF scores of all terms accross all documents
term_importance = df_tfidf.sum(axis=0)

# Sort the terms by importance in descending order
sorted_terms = term_importance.sort_values(ascending=False)

df = pd.DataFrame(sorted_terms)
df.to_csv('sorted-terms.csv')

# print(tfidf_matrix)