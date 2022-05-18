import lda
import numpy as np

from random import random


class SupervisedLDA:
    def __init__(self, n_topics, n_iter=2000, alpha=0.01, eta=0.01, random_state=None,
                 refresh=10):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.eta = eta
        # if random_state is None, check_random_state(None) does nothing
        # other than return the current numpy RandomState
        self.random_state = random_state
        self.refresh = refresh

        if alpha <= 0 or eta <= 0:
            raise ValueError("alpha and eta must be greater than zero")

        # random numbers that are reused
        rng = lda.utils.check_random_state(random_state)
        if random_state:
            random.seed(random_state)
        self._rands = rng.rand(1024 ** 2 // 8)  # 1MiB of random variates

    def fit(self, X, y=None, seed_topics={}, seed_confidence=0):

        self._fit(X, seed_topics=seed_topics, seed_confidence=seed_confidence)
        return self

    def fit_transform(self, X, y=None, seed_topics={}, seed_confidence=0):
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        self._fit(X, seed_topics=seed_topics, seed_confidence=seed_confidence)
        return self.doc_topic_

    def transform(self, X, max_iter=20, tol=1e-16):
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        doc_topic = np.empty((X.shape[0], self.n_topics))
        WS, DS = lda.utils.matrix_to_lists(X)
        # TODO: this loop is parallelizable
        for d in np.unique(DS):
            doc_topic[d] = self._transform_single(WS[DS == d], max_iter, tol)
        return doc_topic

    def _transform_single(self, doc, max_iter, tol):
        PZS = np.zeros((len(doc), self.n_topics))
        for iteration in range(max_iter + 1):  # +1 is for initialization
            PZS_new = self.components_[:, doc].T
            PZS_new *= (PZS.sum(axis=0) - PZS + self.alpha)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis]  # vector to single column matrix
            delta_naive = np.abs(PZS_new - PZS).sum()
            PZS = PZS_new
            if delta_naive < tol:
                break
        theta_doc = PZS.sum(axis=0) / PZS.sum()
        assert len(theta_doc) == self.n_topics
        assert theta_doc.shape == (self.n_topics,)
        return theta_doc

    def _fit(self, X, seed_topics, seed_confidence):
        random_state = lda.utils.check_random_state(self.random_state)
        rands = self._rands.copy()

        self._initialize(X, seed_topics, seed_confidence)
        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                print(f"<{it}> log likelihood: {ll:.0f}")
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
            self._sample_topics(rands)
        ll = self.loglikelihood()
        # note: numpy /= is integer division
        self.components_ = (self.nzw_ + self.eta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_

        self.word_topic_ = (self.nzw_ + self.eta).astype(float)
        self.word_topic_ /= np.sum(self.word_topic_, axis=0)[np.newaxis, :]
        self.word_topic_ = self.word_topic_.T
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        return self

    def _initialize(self, X, seed_topics, seed_confidence):
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.n_topics
        n_iter = self.n_iter

        self.beta = 0.1
        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)  # + self.beta
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)  # + self.alpha
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)  # + W * self.beta

        self.WS, self.DS = WS, DS = lda.utils.matrix_to_lists(X)
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))

        for i in range(N):
            word = WS[i]
            doc = DS[i]

            if word in seed_topics:
                if random() < seed_confidence:
                    topic = seed_topics[word]
                else:
                    topic = i % n_topics
            else:
                topic = i % n_topics
            ZS[i] = topic
            ndz_[doc, topic] += 1
            nzw_[topic, word] += 1
            nz_[topic] += 1

        self.loglikelihoods_ = []

        self.nzw_ = nzw_.astype(np.intc)
        self.ndz_ = ndz_.astype(np.intc)
        self.nz_ = nz_.astype(np.intc)

    def purge_extra_matrices(self):
        del self.topic_word_
        del self.word_topic_
        del self.doc_topic_
        del self.nzw_
        del self.ndz_
        del self.nz_

    def loglikelihood(self):
        nzw, ndz, nz = self.nzw_, self.ndz_, self.nz_
        alpha = self.alpha
        eta = self.eta
        nd = np.sum(ndz, axis=1).astype(np.intc)
        return lda._lda._loglikelihood(nzw, ndz, nz, nd, alpha, eta)

    def _sample_topics(self, rands):
        n_topics, vocab_size = self.nzw_.shape
        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)
        lda._lda._sample_topics(self.WS, self.DS, self.ZS, self.nzw_, self.ndz_, self.nz_,
                                alpha, eta, rands)
