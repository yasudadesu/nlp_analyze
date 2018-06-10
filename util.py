from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter


stopWords = set(stopwords.words('english') + ['using', 'via'])

conference_dict = {
    'ACL': {
        2011: 'http://mirror.aclweb.org/acl2011/accepted_papers.shtml.html',
        2012: 'http://mirror.aclweb.org/acl2012/program/sub00.asp.html',
        2013: 'http://www.acl2013.org/site/accepted-papers.html',
        2014: 'http://acl2014.org/Program.htm',
        2015: 'http://acl2015.org/accepted_papers.html',
        2016: 'http://mirror.aclweb.org/acl2016/indexa779.html?article_id=68',
        2017: 'https://acl2017.wordpress.com/2017/04/05/accepted-papers-and-demonstrations/',
        2018: 'http://acl2018.org/programme/papers/',
    },
    'EMNLP': {
        2011: 'http://conferences.inf.ed.ac.uk/emnlp2011/papers.html',
        2012: 'http://emnlp-conll2012.unige.ch/papers.html',
        2013: 'http://mirror.aclweb.org/emnlp2013/papers.html',
        2014: 'http://emnlp2014.org/papers.html',
        2015: 'http://www.emnlp2015.org/accepted-papers.html',
        2016: 'https://www.aclweb.org/mirror/emnlp2016/accepted-papers.html',
        2017: 'http://emnlp2017.net/accepted-papers.html',
        2018: '',
    },
    'NAACL': {
        2012: '',
        2013: 'http://naacl2013.naacl.org/PapersAccepted.aspx',
        2014: '',
        2015: 'http://naacl.org/naacl-hlt-2015/papers.html',
        2016: 'http://naacl.org/naacl-hlt-2016/accepted_papers.html',
        2017: '',
        2018: 'https://naacl2018.wordpress.com/2018/03/02/list-of-accepted-papers/',
    },
}

def _word_preprocess(word):
    word = word.lower()
    word = word.replace('models', 'model')
    word = word.replace('embeddings', 'embedding')
    word = word.replace('evaluate', 'evaluation')
    word = word.replace('metrics', 'evaluation')
    return word

def text_preprocess(text):
    tokens = word_tokenize(text)
    tokens = list(filter(lambda x: x.lower() not in stopWords, tokens))
    tokens = [_word_preprocess(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token != 'model']
    counter = Counter(tokens)
    return counter