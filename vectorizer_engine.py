# This is more a copied code so not documenting anything
# https://github.com/JRC1995/Abstractive-Summarization
import numpy as np

filename = 'glove.6B.100d.txt'


class VectorizerEngine:

    def __init__(self):
        self.vocab, embd = self.loadGloVe(filename)

        self.embedding = np.asarray(embd)
        self.embedding = self.embedding.astype(np.float32)

        self.word_vec_dim = len(self.embedding[0])
        print("Vectorizer (of dimension {0}) ready and waiting".format(self.word_vec_dim))

    def loadGloVe(self, filename):
        vocab = []
        embd = []
        file = open(filename, 'r')
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
        print('Loaded GloVe!')
        file.close()
        return vocab, embd

    def np_nearest_neighbour(self, x):
        # returns array in embedding that's most similar (in terms of cosine similarity) to x

        xdoty = np.multiply(self.embedding, x)
        xdoty = np.sum(xdoty, 1)
        xlen = np.square(x)
        xlen = np.sum(xlen, 0)
        xlen = np.sqrt(xlen)
        ylen = np.square(self.embedding)
        ylen = np.sum(ylen, 1)
        ylen = np.sqrt(ylen)
        xlenylen = np.multiply(xlen, ylen)
        cosine_similarities = np.divide(xdoty, xlenylen)

        return self.embedding[np.argmax(cosine_similarities)]

    def word2vec(self, word):  # converts a given word into its vector representation
        if word in self.vocab:
            return self.embedding[self.vocab.index(word)]
        else:
            print("word '{0}' not in dictionary".format(word))
            return self.embedding[self.vocab.index('unk')]

    def vec2word(self, vec):  # converts a given vector representation into the represented word
        for x in xrange(0, len(self.embedding)):
            if np.array_equal(self.embedding[x], np.asarray(vec)):
                return self.vocab[x]
        return self.vec2word(self.np_nearest_neighbour(np.asarray(vec)))
