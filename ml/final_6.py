#### 第六题
import json
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report
from scipy.special import digamma
import re

class VariationalMultinomialNB:
    def __init__(self, K=2, alpha=1.0, beta=0.01, max_iter=50):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

    def fit(self, X):
        N, V = X.shape
        K = self.K

        # q(z) 初始化
        gamma = np.random.dirichlet([1]*K, size=N)

        # 变分参数
        alpha_q = np.ones(K) * self.alpha
        lambda_q = np.ones((K, V)) * self.beta

        for _ in range(self.max_iter):
            # ===== M-step =====
            Nk = gamma.sum(axis=0)
            alpha_q = self.alpha + Nk
            lambda_q = self.beta + gamma.T @ X

            # 期望
            E_log_pi = digamma(alpha_q) - digamma(alpha_q.sum())
            E_log_phi = digamma(lambda_q) - digamma(
                lambda_q.sum(axis=1, keepdims=True)
            )

            # ===== E-step =====
            for n in range(N):
                for k in range(K):
                    gamma[n, k] = (
                        E_log_pi[k] + (X[n] * E_log_phi[k]).sum()
                    )

                gamma[n] -= gamma[n].max()
                gamma[n] = np.exp(gamma[n])
                gamma[n] /= gamma[n].sum()

        self.alpha_q = alpha_q
        self.lambda_q = lambda_q

    def predict(self, X):
        E_log_pi = digamma(self.alpha_q) - digamma(self.alpha_q.sum())
        E_log_phi = digamma(self.lambda_q) - digamma(
            self.lambda_q.sum(axis=1, keepdims=True)
        )

        preds = []
        for x in X:
            scores = []
            for k in range(self.K):
                scores.append(E_log_pi[k] + (x * E_log_phi[k]).sum())
            preds.append(np.argmax(scores))

        return np.array(preds)

### load data
with open('ml/data/enron1.json', 'r') as f:
    raw_data = json.load(f)
### tokenize
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
data = []
for d in raw_data:
    words = tokenize(d['text'])
    label = d['label']
    data.append((words, label))
### split train and test data
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=0,
    stratify=[y for _, y in data]
)
word_count = Counter()
for words, _ in train_data:
    word_count.update(words)
filtered_words = [w for w, c in word_count.items() if c >= 5]
vocab = {w: i for i, w in enumerate(filtered_words)}
V = len(vocab)
print("Vocabulary size:", V)
### preprocess data
def build_X(data, vocab):
    N = len(data)
    V = len(vocab)
    X = np.zeros((N, V), dtype=int)
    y = np.zeros(N, dtype=int)

    for i, (words, label) in enumerate(data):
        y[i] = label
        for w in words:
            if w in vocab:
                X[i, vocab[w]] += 1
    return X, y
X_train, y_train = build_X(train_data, vocab)
X_test,  y_test  = build_X(test_data, vocab)
### build model
model = VariationalMultinomialNB(
    K=2,
    alpha=1.0,
    beta=0.06,
    max_iter=30
)
### train
model.fit(X_train)
### predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))






