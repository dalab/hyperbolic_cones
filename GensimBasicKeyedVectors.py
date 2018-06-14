"""
Old version of Gesim's BasicKeyedVectors
"""

from numpy import dot, zeros, dtype, float32 as REAL,\
    double, array, vstack, fromstring, sqrt, newaxis,\
    ndarray, sum as np_sum, prod, ascontiguousarray,\
    argmax
import numpy as np
from six import string_types, iteritems
from six.moves import xrange

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
import logging
logger = logging.getLogger(__name__)



class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class KeyedVectorsBase(utils.SaveLoad):
    """
    Base class to contain vectors and vocab for any set of vectors which are each associated with a key.

    """

    def __init__(self):
        self.syn0 = []
        self.vocab = {}
        self.index2word = []
        self.vector_size = None

    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

         `fname` is the file used to save the vectors in
         `fvocab` is an optional file used to save the vocabulary
         `binary` is an optional boolean indicating whether the data is to be saved
         in binary word2vec format (default: False)
         `total_vec` is an optional parameter to explicitly specify total no. of vectors
         (in case word vectors are appended with document vectors afterwards)

        """
        if total_vec is None:
            total_vec = len(self.vocab)
        vector_size = self.syn0.shape[1]
        if fvocab is not None:
            logger.info("storing vocabulary in %s", fvocab)
            with utils.smart_open(fvocab, 'wb') as vout:
                for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                    vout.write(utils.to_utf8("%s %s\n" % (word, vocab.count)))
        logger.info("storing %sx%s projection weights into %s", total_vec, vector_size, fname)
        assert (len(self.vocab), vector_size) == self.syn0.shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
            # store in sorted order: most frequent words at the top
            for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                row = self.syn0[vocab.index]
                if binary:
                    fout.write(utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                             limit=None, datatype=REAL):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        `norm_only` is a boolean indicating whether to only store normalised word2vec vectors in memory.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).

        If you trained the C model using non-utf8 encoding for words, specify that
        encoding in `encoding`.

        `unicode_errors`, default 'strict', is a string suitable to be passed as the `errors`
        argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source
        file may include word tokens truncated in the middle of a multibyte unicode character
        (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.

        `limit` sets a maximum number of word-vectors to read from the file. The default,
        None, means read all.

        `datatype` (experimental) can coerce dimensions to a non-default float type (such
        as np.float16) to save memory. (Such types may result in much slower bulk operations
        or incompatibility with optimized routines.)

        """
        counts = None
        if fvocab is not None:
            logger.info("loading word counts from %s", fvocab)
            counts = {}
            with utils.smart_open(fvocab) as fin:
                for line in fin:
                    word, count = utils.to_unicode(line).strip().split()
                    counts[word] = int(count)

        logger.info("loading projection weights from %s", fname)
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format
            if limit:
                vocab_size = min(vocab_size, limit)
            result = cls()
            result.vector_size = vector_size
            result.syn0 = zeros((vocab_size, vector_size), dtype=datatype)

            def add_word(word, weights):
                word_id = len(result.vocab)
                if word in result.vocab:
                    logger.warning("duplicate word '%s' in %s, ignoring all but first", word, fname)
                    return
                if counts is None:
                    # most common scenario: no vocab file given. just make up some bogus counts, in descending order
                    result.vocab[word] = Vocab(index=word_id, count=vocab_size - word_id)
                elif word in counts:
                    # use count from the vocab file
                    result.vocab[word] = Vocab(index=word_id, count=counts[word])
                else:
                    # vocab file given, but word is missing -- set count to None (TODO: or raise?)
                    logger.warning("vocabulary file is incomplete: '%s' is missing", word)
                    result.vocab[word] = Vocab(index=word_id, count=None)
                result.syn0[word_id] = weights
                result.index2word.append(word)

            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                for _ in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch == b'':
                            raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = fromstring(fin.read(binary_len), dtype=REAL)
                    add_word(word, weights)
            else:
                for line_no in xrange(vocab_size):
                    line = fin.readline()
                    if line == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                    word, weights = parts[0], [REAL(x) for x in parts[1:]]
                    add_word(word, weights)
        if result.syn0.shape[0] != len(result.vocab):
            logger.info(
                "duplicate words detected, shrinking matrix size from %i to %i",
                result.syn0.shape[0], len(result.vocab)
            )
            result.syn0 = ascontiguousarray(result.syn0[: len(result.vocab)])
        assert (len(result.vocab), vector_size) == result.syn0.shape

        logger.info("loaded %s matrix from %s", result.syn0.shape, fname)
        return result

    def similarity(self, w1, w2):
        """
        Compute similarity between vectors of two input words.
        To be implemented by child class.

        """
        raise NotImplementedError

    def distance(self, w1, w2):
        """
        Compute distance between vectors of two input words.
        To be implemented by child class.

        """
        raise NotImplementedError

    def distances(self, word_or_vector, other_words=()):
        """
        Compute distances from given word or vector to all words in `other_words`.
        If `other_words` is empty, return distance between `word_or_vectors` and all words in vocab.
        To be implemented by child class.

        """
        raise NotImplementedError

    def word_vec(self, word):
        """
        Accept a single word as input.
        Returns the word's representations in vector space, as a 1D numpy array.

        Example::

          >>> trained_model.word_vec('office')
          array([ -1.40128313e-02, ...])

        """
        if word in self.vocab:
            result = self.syn0[self.vocab[word].index]
            result.setflags(write=False)
            return result
        else:
            raise KeyError("word '%s' not in vocabulary" % word)

    def __getitem__(self, words):
        """
        Accept a single word or a list of words as input.

        If a single word: returns the word's representations in vector space, as
        a 1D numpy array.

        Multiple words: return the words' representations in vector space, as a
        2d numpy array: #words x #vector_size. Matrix rows are in the same order
        as in input.

        Example::

          >>> trained_model['office']
          array([ -1.40128313e-02, ...])

          >>> trained_model[['office', 'products']]
          array([ -1.40128313e-02, ...]
                [ -1.70425311e-03, ...]
                 ...)

        """
        if isinstance(words, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.word_vec(words)

        return vstack([self.word_vec(word) for word in words])

    def __contains__(self, word):
        return word in self.vocab

    def most_similar_to_given(self, w1, word_list):
        """Return the word from word_list most similar to w1.

        Args:
            w1 (str): a word
            word_list (list): list of words containing a word most similar to w1

        Returns:
            the word in word_list with the highest similarity to w1

        Raises:
            KeyError: If w1 or any word in word_list is not in the vocabulary

        Example::

          >>> trained_model.most_similar_to_given('music', ['water', 'sound', 'backpack', 'mouse'])
          'sound'

          >>> trained_model.most_similar_to_given('snake', ['food', 'pencil', 'animal', 'phone'])
          'animal'

        """
        return word_list[argmax([self.similarity(w1, word) for word in word_list])]

    def words_closer_than(self, w1, w2):
        """
        Returns all words that are closer to `w1` than `w2` is to `w1`.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        list (str)
            List of words that are closer to `w1` than `w2` is to `w1`.

        Examples
        --------

        >>> model.words_closer_than('carnivore.n.01', 'mammal.n.01')
        ['dog.n.01', 'canine.n.02']

        """
        all_distances = self.distances(w1)
        w1_index = self.vocab[w1].index
        w2_index = self.vocab[w2].index
        closer_node_indices = np.where(all_distances < all_distances[w2_index])[0]
        return [self.index2word[index] for index in closer_node_indices if index != w1_index]

    def rank(self, w1, w2):
        """
        Rank of the distance of `w2` from `w1`, in relation to distances of all words from `w1`.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        int
            Rank of `w2` from `w1` in relation to all other nodes.

        Examples
        --------

        >>> model.rank('mammal.n.01', 'carnivore.n.01')
        3

        """
        return len(self.words_closer_than(w1, w2)) + 1

