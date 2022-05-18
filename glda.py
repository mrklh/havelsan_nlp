import pickle
from operator import itemgetter

import pandas as pd
import pyLDAvis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from yellowbrick.text import FreqDistVisualizer
import numpy as np
import re
from sklearn.model_selection import KFold
import matplotlib as mpl
from sklearn.metrics import top_k_accuracy_score

from constants import AVAILABLE_STOP_WORDS, BLACK_LIST, NLTK_STOP_WORDS, APRIORI_LIST

mpl.use("TkAgg")
from supervised_lda import SupervisedLDA

KFOLD = True
TOPIC_NUMBER = 5
NITER = 50
ALPHA = .01
ETA = .9
CONF = 1
IN_OR_OUT = 1
AVAILABLE_STOP_WORDS.extend(NLTK_STOP_WORDS)
AVAILABLE_STOP_WORDS.extend(BLACK_LIST)

topics = ['economics', 'health', 'life', 'sports', 'technology']
topic_labels = [
    f"__label__{x}" for x in topics
]


def measure_performance(results, test_df, model, test_tokens, print_error_samples):
    y_pred = np.fliplr(results.argsort())
    y_true = test_df['label'].apply(lambda x: topic_labels.index(x)).tolist()
    # print(np.isnan(y_true).sum())

    for i in range(1, 4):
        print(top_k_accuracy_score(y_true, results, k=i))

    aa = confusion_matrix(y_pred[:, 0], y_true)
    print(aa)

    if print_error_samples:
        for i, y in enumerate(y_true):
            if y_true[i] == 2 and y_pred[i, 0] == 0:
                print(i, test_df.iloc[i]['pure_text'])
                top_n_topics = itemgetter(*results[i].argsort()[::-1])(topics)
                top_n_weights = itemgetter(*results[i].argsort()[::-1])(results[i])
                print(list(zip(top_n_topics, top_n_weights)))

                word_idx = test_tokens[i, :].nonzero()[1].tolist()
                for w_index in word_idx:
                    word_dist = np.array(model.word_topic_[w_index])
                    top_tops = [topics[x] for x in word_dist.argsort()[:-4:-1]]
                    word_dist.sort()
                    top_weights = word_dist[:-4:-1]
                    word = tf_feature_names[w_index]
                    category_details = [[f"{x} ({y:.5})" for x, y in zip(top_tops, top_weights)] + list(top_weights)]
                    print(f"{word}: {category_details}")


def process_text(word):
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\'|’)', '', word)
    word = re.sub(r'\d+(.\d+)*', ' <number> ', word)
    word = word.lower()
    return word


def get_top_n_words(tfidf, tfidf_vectorizer, n_top=20):
    importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
    tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names())
    return tfidf_feature_names[importance[:n_top]]


def load_and_preprocess():
    data = []
    for topic in topics:
        with open(f"data/{topic}.txt", "r", encoding='utf8') as f:
            lines = f.readlines()
            data.extend(lines)

    df = pd.DataFrame(columns=['text'], data=data)
    df = df.sample(frac=.5).reset_index(drop=True)
    df['label'] = df['text'].apply(lambda x: x[:x.index(' ')])
    df['pure_text'] = df['text'].apply(lambda x: x[x.index(' ') + 1:])
    df['processed_text'] = df['pure_text'].apply(lambda x: process_text(x))

    return df


def create_vocab(df):
    token_vectorizer = CountVectorizer(min_df=10,
                                       max_df=.65,
                                       ngram_range=(1, 3),
                                       stop_words=AVAILABLE_STOP_WORDS)
    X = token_vectorizer.fit_transform(df['processed_text'])
    tf_feature_names = token_vectorizer.get_feature_names()

    word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))

    seed_topics = {}

    for t_id, topic_name in enumerate(topics):
        print(topic_name)
        vocab_list = APRIORI_LIST[topic_name]
        for word in vocab_list:
            if word not in word2id:
                continue
            seed_topics[word2id[word]] = t_id

    return X, token_vectorizer, word2id, tf_feature_names, seed_topics


def visualize(X, topic_word, tf_feature_names, model):
    # K-fold kth fold's word frequency plot
    visualizer = FreqDistVisualizer(features=tf_feature_names, orient='v')
    visualizer.fit(X)
    visualizer.show()

    # Topics' top n words with probs
    n_top_words = 50
    for i, topic_dist in enumerate(topic_word):
        topics_words = [
            f"{x} ({y:.3f})" for x, y in
            zip(
                np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words + 1):-1],
                np.sort(topic_dist)[:-(n_top_words + 1):-1]
            )
        ]
        print('Topic {} ({}): {}'.format(i, topics[i], ' | '.join(topics_words)))

    doc_lengths = X.sum(axis=1).ravel().tolist()[0]

    # transpose the dtm and get a sum of the overall term frequency
    dtm_trans = X.T
    term_frequencies = dtm_trans.sum(axis=1).ravel().tolist()[0]

    data = {'topic_term_dists': model.topic_word_, 'doc_topic_dists': model.doc_topic_, 'doc_lengths': doc_lengths,
            'vocab': tf_feature_names, 'term_frequency': term_frequencies}
    tef_vis_data = pyLDAvis.prepare(**data, sort_topics=False)

    pyLDAvis.display(tef_vis_data)
    pyLDAvis.save_html(tef_vis_data, "figs/LDAvis.html")


def train(X, seed_topics):
    model = SupervisedLDA(
        n_topics=TOPIC_NUMBER,
        n_iter=NITER,
        alpha=ALPHA,
        eta=ETA
    )
    model.fit(X, seed_topics=seed_topics, seed_confidence=.9)

    return model


def predict(df, model, print_error_samples=False):
    test_tokens = token_vectorizer.transform(df['processed_text'])
    results = model.transform(test_tokens)
    measure_performance(results, df, model, test_tokens, print_error_samples)
    return results


def predict_single(text, model=None, token_vectorizer=None):
    p_text = process_text(text)
    if not model or not token_vectorizer:
        with open('lda_model_v1.pkl', 'rb') as f:
            load = pickle.load(f)
            token_vectorizer = load['vectorizer']
            model = load['model']

    test_tokens = token_vectorizer.transform([p_text])
    results = model.transform(test_tokens)
    topic_order = np.fliplr(results.argsort())
    selected_topic = topics[topic_order[0][0]]
    print("------------------- Text --------------------")
    print(text[:-1])
    print("------------- Selected Topic ----------------")
    print(f"{selected_topic}")
    print("------------- Topic Weights -----------------")
    top_n_topics = itemgetter(*results[0].argsort()[::-1])(topics)
    top_n_weights = itemgetter(*results[0].argsort()[::-1])(results[0])
    for topic, score in list(zip(top_n_topics, top_n_weights)):
        print(f"{topic}: {score}")

    print("------------Most Effective Words ------------")
    word_idx = test_tokens[0, :].nonzero()[1].tolist()
    for w_index in word_idx:
        word_dist = np.array(model.word_topic_[w_index])
        top_tops = [topics[x] for x in word_dist.argsort()[:-4:-1]]
        word_dist.sort()
        top_weights = word_dist[:-4:-1]
        word = tf_feature_names[w_index]
        category_details = [[f"{x} ({y:.5})" for x, y in zip(top_tops, top_weights)]]
        if selected_topic == top_tops[0] and top_weights[0] > .005:
            print(f"{word}: {category_details}")


df = load_and_preprocess()
kf = KFold(n_splits=5, shuffle=True, random_state=2)

for train_index, test_index in kf.split(df):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    X_train, token_vectorizer, word2id, tf_feature_names, seed_topics = create_vocab(df)
    model = train(X_train, seed_topics)
    visualize(X=X_train, topic_word=model.topic_word_, tf_feature_names=tf_feature_names, model=model)

    y_pred = predict(test_df, model, print_error_samples=True)

with open('lda_model_v1.pkl', 'wb') as f:
    pickle.dump({'vectorizer': token_vectorizer, 'model': model}, f)


predict_single('Bu sezon şampiyonluk çok erken belli oldu.')