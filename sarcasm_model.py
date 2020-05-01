import re
from nltk.tokenize import TweetTokenizer
import itertools
import time
import numpy
from keras.models import model_from_json
from collections import defaultdict
import tensorflow as tf


def normalize_word(word):
    temp = word
    while True:
        w = re.sub(r"([a-zA-Z])\1\1", r"\1\1", temp)
        if w == temp:
            break
        else:
            temp = w
    return w


def split_hashtags(term, wordlist, split_word_list, dump_file=''):
    # print('term::',term)

    if len(term.strip()) == 1:
        return ['']

    if split_word_list != None and term.lower() in split_word_list:
        # print('found')
        return split_word_list.get(term.lower()).split(' ')
    else:
        print(term)

    # discarding # if exists
    if (term.startswith('#')):
        term = term[1:]

    if (wordlist != None and term.lower() in wordlist):
        return [term.lower()]

    words = []
    # max freq
    penalty = -69971
    max_coverage = penalty

    split_words_count = 6
    # checking camel cases
    term = re.sub(r'([0-9]+)', r' \1', term)
    term = re.sub(r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|0th)', r'\1 ', term)
    term = re.sub(r'([A-Z][^A-Z ]+)', r' \1', term.strip())
    term = re.sub(r'([A-Z]{2,})+', r' \1', term)
    words = term.strip().split(' ')

    n_splits = 0

    if (len(words) < 3):
        # splitting lower case and uppercase words upto 5 words
        chars = [c for c in term.lower()]

        found_all_words = False

        while (n_splits < split_words_count and not found_all_words):
            for idx in itertools.combinations(range(0, len(chars)), n_splits):
                output = numpy.split(chars, idx)
                line = [''.join(o) for o in output]

                score = (1. / len(line)) * sum(
                    [wordlist.get(
                        word.strip()) if word.strip() in wordlist else 0. if word.strip().isnumeric() else penalty for
                     word in line])

                if (score > max_coverage):
                    words = line
                    max_coverage = score

                    line_is_valid_word = [word.strip() in wordlist if not word.isnumeric() else True for word in line]

                    if (all(line_is_valid_word)):
                        found_all_words = True

                    # uncomment to debug hashtag splitting
                    # print(line, score, line_is_valid_word)

            n_splits = n_splits + 1

    # removing hashtag sign
    words = [str(s) for s in words]

    # dumping splits for debug
    with open(dump_file, 'a') as f:
        if (term != '' and len(words) > 0):
            f.write('#' + str(term).strip() + '\t' + ' '.join(words) + '\t' + str(n_splits) + '\n')

    return words


def filter_text(text, word_list, split_word_list, emoji_dict, abbreviation_dict, normalize_text=False,
                split_hashtag=False,
                ignore_profiles=False,
                replace_emoji=True):
    filtered_text = []

    filter_list = ['/', '-', '=', '+', '…', '\\', '(', ')', '&', ':']

    for t in text:
        word_tokens = None

        # discarding symbols
        # if (str(t).lower() in filter_list):
        #     continue

        # ignoring profile information if ignore_profiles is set
        if (ignore_profiles and str(t).startswith("@")):
            continue

        # ignoring links
        if (str(t).startswith('http')):
            continue

        # ignoring sarcastic marker
        # uncomment the following line for Fracking sarcasm using neural network
        # if (str(t).lower() in ['#sarcasm','#sarcastic', '#yeahright','#not']):
        #     continue

        # for onlinesarcasm
        # comment if you are running the code for Fracking sarcasm using neural network
        if (str(t).lower() in ['#sarcasm']):
            continue

        # replacing emoji with its unicode description
        if (replace_emoji):
            if (t in emoji_dict):
                t = emoji_dict.get(t).split('_')
                filtered_text.extend(t)
                continue

        # splitting hastags
        if (split_hashtag and str(t).startswith("#")):
            splits = split_hashtags(t, word_list, split_word_list, dump_file='../resource/hastash_split_dump.txt')
            # adding the hashtags
            if (splits != None):
                filtered_text.extend([s for s in splits if (not filtered_text.__contains__(s))])
                continue

        # removes repeatation of letters
        if (normalize_text):
            t = normalize_word(t)

        # expands the abbreviation
        if (t in abbreviation_dict):
            tokens = abbreviation_dict.get(t).split(' ')
            filtered_text.extend(tokens)
            continue

        # appends the text
        filtered_text.append(t)

    return filtered_text


def parsedata(lines, word_list, split_word_list, emoji_dict, abbreviation_dict, normalize_text=False,
              split_hashtag=False,
              ignore_profiles=False,
              lowercase=False, replace_emoji=True, n_grams=None, at_character=False):
    data = []
    for i, line in enumerate(lines):
        # if (i % 100 == 0):
        #     print(str(i) + '...', end='', flush=True)

        try:

            # convert the line to lowercase
            if (lowercase):
                # print(i)
                line = line.lower()

            # split into token
            token = line.split('\t')
            # print(token)
            # ID
            id = token[0]

            # label
            label = int(token[1].strip())

            # tweet text
            target_text = TweetTokenizer().tokenize(token[2].strip())
            # print(target_text)
            if (at_character):
                target_text = [c for c in token[2].strip()]

            if (n_grams != None):
                n_grams_list = list(create_ngram_set(target_text, ngram_value=n_grams))
                target_text.extend(['_'.join(n) for n in n_grams_list])

            # filter text
            target_text = filter_text(target_text, word_list, split_word_list, emoji_dict, abbreviation_dict,
                                      normalize_text,
                                      split_hashtag,
                                      ignore_profiles, replace_emoji=replace_emoji)

            # awc dimensions
            dimensions = []
            if (len(token) > 3 and token[3].strip() != 'NA'):
                dimensions = [dimension.split('@@')[1] for dimension in token[3].strip().split('|')]

            # context tweet
            context = []
            if (len(token) > 4):
                if (token[4] != 'NA'):
                    context = TweetTokenizer().tokenize(token[4].strip())
                    context = filter_text(context, word_list, split_word_list, emoji_dict, abbreviation_dict,
                                          normalize_text,
                                          split_hashtag,
                                          ignore_profiles, replace_emoji=replace_emoji)

            # author
            author = 'NA'
            if (len(token) > 5):
                author = token[5]

            if (len(target_text) != 0):
                # print((label, target_text, dimensions, context, author))
                data.append((id, label, target_text, dimensions, context, author))
        except:
            raise
    print('')
    return data


def load_resources(word_file_path, split_word_path, emoji_file_path, split_hashtag=False, replace_emoji=True):
    word_list = None
    emoji_dict = None

    # load split files
    split_word_list = load_split_word(split_word_path)
    # print(split_word_list)

    # load word dictionary
    if (split_hashtag):
        word_list = InitializeWords(word_file_path)
        # print(word_list)

    if (replace_emoji):
        emoji_dict = load_unicode_mapping(emoji_file_path)
        # print(emoji_dict)

    abbreviation_dict = load_abbreviation()

    return word_list, emoji_dict, split_word_list, abbreviation_dict


def loaddata(filename, word_file_path, split_word_path, emoji_file_path, normalize_text=False, split_hashtag=False,
             ignore_profiles=False,
             lowercase=True, replace_emoji=True, n_grams=None, at_character=False):
    word_list, emoji_dict, split_word_list, abbreviation_dict = load_resources(word_file_path, split_word_path,
                                                                               emoji_file_path,
                                                                               split_hashtag=split_hashtag,
                                                                               replace_emoji=replace_emoji)
    # lines = open(filename, 'r').readlines()

    line = ['headline\t0\t'+filename+'\n']
    # print(f'lines : {lines}')
    # print(f'word_file_path : {word_file_path}')

    data = parsedata(line, word_list, split_word_list, emoji_dict, abbreviation_dict, normalize_text=normalize_text,
                     split_hashtag=split_hashtag,
                     ignore_profiles=ignore_profiles, lowercase=lowercase, replace_emoji=replace_emoji,
                     n_grams=n_grams, at_character=at_character)
    # print(f'data : ${data}')
    return data


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def load_split_word(split_word_file_path):
    split_word_dictionary = defaultdict()
    with open(split_word_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.lower().strip().split('\t')
            if (len(tokens) >= 2):
                split_word_dictionary[tokens[0]] = tokens[1]

    # print('split entry found:', len(split_word_dictionary.keys()))
    return split_word_dictionary


def InitializeWords(word_file_path):
    word_dictionary = defaultdict()

    with open(word_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.lower().strip().split('\t')
            word_dictionary[tokens[0]] = int(tokens[1])

    for alphabet in "abcdefghjklmnopqrstuvwxyz":
        if (alphabet in word_dictionary):
            word_dictionary.__delitem__(alphabet)

    for word in ['ann', 'assis',
                 'bz',
                 'ch', 'cre', 'ct',
                 'di',
                 'ed', 'ee',
                 'ic',
                 'le',
                 'ng', 'ns',
                 'pr', 'picon',
                 'th', 'tle', 'tl', 'tr',
                 'um',
                 've',
                 'yi'
                 ]:
        if (word in word_dictionary):
            word_dictionary.__delitem__(word)

    return word_dictionary


def load_unicode_mapping(path):
    emoji_dict = defaultdict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split('\t')
            emoji_dict[tokens[0]] = tokens[1]
    return emoji_dict


def load_abbreviation(path='./resource/abbreviations.txt'):
    abbreviation_dict = defaultdict()
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            token = line.lower().strip().split('\t')
            abbreviation_dict[token[0]] = token[1]
    return abbreviation_dict


def vectorize_word_dimension(data, vocab, drop_dimension_index=None, verbose=False):
    X = []
    Y = []
    D = []
    C = []
    A = []

    known_words_set = set()
    unknown_words_set = set()

    tokens = 0
    token_coverage = 0

    for id, label, line, dimensions, context, author in data:
        vec = []
        context_vec = []
        if (len(dimensions) != 0):
            dvec = [vocab.get(d) for d in dimensions]
        else:
            dvec = [vocab.get('unk')] * 11

        if drop_dimension_index != None:
            dvec.pop(drop_dimension_index)

        # tweet
        for words in line:
            tokens = tokens + 1
            if (words in vocab):
                vec.append(vocab[words])
                token_coverage = token_coverage + 1
                known_words_set.add(words)
            else:
                vec.append(vocab['unk'])
                unknown_words_set.add(words)
        # context_tweet
        if (len(context) != 0):
            for words in line:
                tokens = tokens + 1
                if (words in vocab):
                    context_vec.append(vocab[words])
                    token_coverage = token_coverage + 1
                    known_words_set.add(words)
                else:
                    context_vec.append(vocab['unk'])
                    unknown_words_set.add(words)
        else:
            context_vec = [vocab['unk']]

        X.append(vec)
        Y.append(label)
        D.append(dvec)
        C.append(context_vec)
        A.append(author)

    if verbose:
        print('Token coverage:', token_coverage / float(tokens))
        print('Word coverage:', len(known_words_set) / float(len(vocab.keys())))

    return numpy.asarray(X), numpy.asarray(Y), numpy.asarray(D), numpy.asarray(C), numpy.asarray(A)


def pad_sequence_1d(sequences, maxlen=None, dtype='float32', padding='pre', truncating='pre', value=0.):
    X = [vectors for vectors in sequences]

    nb_samples = len(X)

    x = (numpy.zeros((nb_samples, maxlen)) * value).astype(dtype)

    for idx, s in enumerate(X):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)

    return x


class sarcasm_model():
    _train_file = None
    _test_file = None
    _tweet_file = None
    _output_file = None
    _model_file_path = None
    _word_file_path = None
    _split_word_file_path = None
    _emoji_file_path = None
    _vocab_file_path = None
    _input_weight_file_path = None
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._line_maxlen = 30


class test_model():
    # print(1)
    test = None
    model = None
    graph = tf.get_default_graph()

    def __init__(self, input_weight_file_path=None):
        # print(2)
        print('initializing...')
        self._line_maxlen = 30
        self._model_file_path = './resource/text_model/weights/'
        self._word_file_path = './resource/word_list_freq.txt'
        self._split_word_file_path = './resource/word_split.txt'
        self._emoji_file_path = './resource/emoji_unicode_names_final.txt'
        self._vocab_file_path = './resource/text_model/vocab_list.txt'
        self._output_file = './resource/text_model/TestResults1.txt'
        self._input_weight_file_path = input_weight_file_path

        # print('test_maxlen', self._line_maxlen)

    def load_trained_model(self, model_file='model.json', weight_file='model.json.hdf5'):
        # print(3)
        start = time.time()
        self.__load_model(self._model_file_path + model_file, self._model_file_path + weight_file)
        end = time.time()
        print('model loading time::', (end - start))

    def __load_model(self, model_path, model_weight_path):
        # print(4)
        self.model = model_from_json(open(model_path).read())
        print('model loaded from file...')
        self.model.load_weights(model_weight_path)
        print('model weights loaded from file...')

    def load_vocab(self):
        # print(5)
        vocab = defaultdict()
        with open(self._vocab_file_path, 'r') as f:
            for line in f.readlines():
                key, value = line.split('\t')
                vocab[key] = value

        return vocab

    def predict(self, test_file, verbose=False):
        # print(test_file)
        # label = []

        try:
            start = time.time()
            self.test = loaddata(test_file, self._word_file_path, self._split_word_file_path, self._emoji_file_path,
                                 normalize_text=True, split_hashtag=True,
                                 ignore_profiles=False)
            end = time.time()
            # print(self.test)
            if (verbose == True):
                print('test resource loading time::', (end - start))

            self._vocab = self.load_vocab()
            # print('vocab loaded...')

            start = time.time()
            tX, tY, tD, tC, tA = vectorize_word_dimension(self.test, self._vocab)
            tX = pad_sequence_1d(tX, maxlen=self._line_maxlen)
            end = time.time()
            if (verbose == True):
                print('test resource preparation time::', (end - start))

            with self.graph.as_default():
                prediction_probability = self.model.predict(tX, batch_size=1, verbose=1)
            # print(prediction_probability[0])
            return prediction_probability[0]
        except Exception as e:
            print('Error:', e)
            raise


# if __name__ == "__main__":
#     t = test_model()
#     t.load_trained_model()
#     t.predict('I loovee when people text back ... 😒 #sarcastictweet')
