
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from .preprocessing import tokenize_and_filter

# Convert text data into TF-IDF representation
def topic_modelling(text):

    tokens = tokenize_and_filter(text)
    document = ' '.join(tokens)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # max_features limits the number of terms
    X = vectorizer.fit_transform([document])

    # Apply LDA (Latent Dirichlet Allocation) to discover topics
    lda_model = LatentDirichletAllocation(n_components=3, random_state=42)  # n_components = number of topics
    lda_model.fit(X)

    # Print the topics
    n_top_words = 4    # Number of top words per topic
    feature_names = vectorizer.get_feature_names_out()

    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()
    print(f"Perplexity: {lda_model.perplexity(X)}")

if __name__ == '__main__':
    user_message = "I love programming in Python. Python is a great programming language. I love data science and machine learning. Data science is amazing."
    result = topic_modelling(user_message)
