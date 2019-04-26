import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(classes, data):
    vectorizer = TfidfVectorizer()
    labels = np.array([classes[i0] for i0 in data["labels"]])
    features = vectorizer.fit_transform(data["text"].values.astype('U')).toarray()
    return features, labels
