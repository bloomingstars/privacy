from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm
nlp = en_core_web_sm.load()

class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        keys=[]
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            #print(key + ' - ' + str(value))
            keys.append(key)
            if i > number:
                break
        print(keys)
        return keys
        
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN','VERB'], 
                window_size=2, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight

import nltk
import re
from nltk.corpus import stopwords
import numpy as numpy
import nltk 
from nltk.corpus import wordnet
stop_words=set(stopwords.words('english'))
stop_words.add('notice')
sentences=[]
bag_of_vectors=[]
keywords=[]
def read(filename):
    file = open(filename, "r")
    for line in file.readlines():
        sentences.append(line)
        
def tokenize(sentence):
    words = re.sub("[^a-zA-Z]", " ",  sentence).split()
    cleaned_text = [w.lower() for w in words if w not in stop_words]
    #print(cleaned_text)
    return sorted(list(set(cleaned_text)))

def tokenize_and_pos(sent):
    wordsList = nltk.word_tokenize(sent)
    lsit=nltk.pos_tag(wordsList)
    print('list:: ',lsit)
    newlist=list()
    for tup in lsit:
        ste=set()
        if(tup[1]=='NN' or tup[1]=='NNP'):
            ste.add(tup[0])
            ste.add('n')

        elif(tup[1]=='VB'):
            ste.add(tup[0])
            ste.add('v')

        if(len(ste) != 0):
            newlist.append(ste)

    return newlist

def pos(w):
    syn = wordnet.synsets(w)[0]
    return syn.name()

def generate_bow(sent):
    vocab = tokenize_and_pos(sent)
    print(vocab)
    print("Word List for Document \n{0} \n".format(vocab));
    for sentence in sentences:
        words = getkey(sentence)
        sim=0;
        maxset=[0,'']
        for w in words:
            for word,i in (vocab):
                print('word: ',word+' i: ',i,' w ',w)
                if(isinstance(w, list)):
                    for w1 in w:
                        if wordnet.synset(pos(word)).wup_similarity(wordnet.synset(pos(w1))) is not None:
                            sim+=wordnet.synset(pos(word)).wup_similarity(wordnet.synset(pos(w1)))
                else:
                    if wordnet.synset(pos(word)).wup_similarity(wordnet.synset(pos(w))) is not None:
                         sim+=wordnet.synset(pos(word)).wup_similarity(wordnet.synset(pos(w)))
        if sim>maxset[0]:
            maxset=[ sim, sentence]
        print("{0}\n{1}\n".format(sentence, sim))
        return maxset;

def getkey(text):
    keywords=[]
    tr4w = TextRank4Keyword()
    tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
    keys=tr4w.get_keywords(2)
    for key in keys:
        if key is not None:
            keywords.append(key)
            keywords.append(create_synonyms(key))
    print(keywords)
    return keywords
    
def create_synonyms(w):
    synonyms = []
    for syn in wordnet.synsets(w):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

read('G:\Privacy\hw5\checklist.txt')
generate_bow('Not Provides notice on the Web site of what information it collects from children')
file = open('G:\Privacy\hw5\checklist.txt', "r")
for line in file.readlines():
    generate_bow(line);
