# The Ward’s minimum variance method as our linkage criterion to minimize total within-cluster variance
# at each step, we find the pair of clusters that leads to the minimum increase in total within-cluster variance after merging.

# Normalize corpus bu removing stop words, special characters, white space and tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
# tokenize using regex schema
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
norm_corpus = np.array(['sky blue beautiful', 'love blue beautiful sky',
       'quick brown fox jumps lazy dog',
       'kings breakfast sausages ham bacon eggs toast beans',
       'love green eggs ham sausages bacon',
       'brown fox quick blue dog lazy', 'sky blue sky beautiful today',
       'dog lazy brown fox quick'])
def normalize_document(doc):
    # lowercase and remove special characters\whitespace^M
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, flags=re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered token
    doc = ' '.join(filtered_tokens)
    return doc
# Let function include numpy input
normalize_corpus = np.vectorize(normalize_document)
# tf-idf

tv = TfidfVectorizer(min_df=0., max_df=1., norm="l2",
                     use_idf=True, smooth_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()
vocab = tv.get_feature_names()

# similar features using cosine metric

similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)

# Hierarchical clustering using Ward linkage

Z = linkage(similarity_matrix, 'ward')
pd.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2',
                         'Distance', 'Cluster Size'], dtype="object")

# Visualization

plt.figure(figsize=(8, 3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=1.0, c="k", ls="--", lw=0.5)

from scipy.cluster.hierarchy import fcluster
max_dist = 1.0
cluster_labels = fcluster(Z, max_dist, criterion="distance")
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
pd.concat([corpus_df, cluster_labels], axis=1)