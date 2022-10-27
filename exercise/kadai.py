# %%
from chapter01 import get_string_from_file
from chapter02 import create_wordcloud, get_japanese_fonts

#%matplotlib inline

# %%
from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSKeepFilter

string = 'この研究室は猫に支配されている。'
keep_pos = ['名詞']
analyzer = Analyzer(token_filters=[POSKeepFilter(keep_pos),
                                   ExtractAttributeFilter('surface')])
print(list(analyzer.analyze(string)))


# %%
def get_words(string, keep_pos=None):
    filters = []
    if keep_pos is None:
        filters.append(POSStopFilter(['記号']))       # 記号を除外
    else:
        filters.append(POSKeepFilter(keep_pos))       # 指定品詞を抽出
    filters.append(ExtractAttributeFilter('surface'))
    a = Analyzer(token_filters=filters)               # 後処理を指定
    return list(a.analyze(string))

from collections import Counter

string = get_string_from_file('rasyomon.txt')
words = get_words(string, keep_pos=['名詞'])
count = Counter(words)
font = get_japanese_fonts()[0]
create_wordcloud(count, font)

# %%
# Listing 3.6 #

from gensim import corpora

# %%
# Listing 3.9 #

def build_corpus(file_list, dic_file=None, corpus_file=None):
    docs = []
    for f in file_list:
        text = get_string_from_file(f)
        words = get_words(text, keep_pos=['名詞'])
        docs.append(words)
        # ファイル名を表示
        print(f)
    dic = corpora.Dictionary(docs)
    if not (dic_file is None):
        dic.save(dic_file)
    bows = [dic.doc2bow(d) for d in docs]
    if not (corpus_file is None):
        corpora.MmCorpus.serialize(corpus_file, bows)
    return dic, bows



# %%
# Listing 3.10 #

def bows_to_cfs(bows):
    cfs = dict()
    for b in bows:
        for id, f in b:
            if not id in cfs:
                cfs[id] = 0
            cfs[id] += int(f)
    return cfs

def load_dictionary_and_corpus(dic_file, corpus_file):
    dic = corpora.Dictionary.load(dic_file)
    bows = list(corpora.MmCorpus(corpus_file))
    if not hasattr(dic, 'cfs'):
        dic.cfs = bows_to_cfs(bows)
    return dic, bows


# %%Listing 3.11 #

from gensim import models


# %%
# Listing 3.12 #

def load_aozora_corpus():
    return load_dictionary_and_corpus('data/aozora/aozora.dic',
                                      'data/aozora/aozora.mm')

def get_bows(texts, dic, allow_update=False):
    bows = []
    for text in texts:
        words = get_words(text, keep_pos=['名詞'])
        bow = dic.doc2bow(words, allow_update=allow_update)
        bows.append(bow)
    return bows

import copy

def add_to_corpus(texts, dic, bows, replicate=False):
    if replicate:
        dic = copy.copy(dic)
        bows = copy.copy(bows)
    texts_bows = get_bows(texts, dic, allow_update=True)
    bows.extend(texts_bows)
    return dic, bows, texts_bows


# %%
# Listing 3.13 #

def get_weights(bows, dic, model, surface=False, N=1000):
    # TF・IDFを計算
    weights = model[bows]
    # TF・IDFの値を基準に降順にソート．最大でN個を抽出
    weights = [sorted(w,key=lambda x:x[1], reverse=True)[:N] for w in weights]
    if surface:
        return [[(dic[x[0]], x[1]) for x in w] for w in weights]
    else:
        return weights


# %%
dic, bows = load_aozora_corpus()
melos_text = get_string_from_file('rasyomon.txt')

dic, bows, melos_bows = add_to_corpus([melos_text], dic, bows)
tfidf_model = models.TfidfModel(bows, normalize=True)
weights = get_weights(melos_bows, dic, tfidf_model, surface=True)
count = dict(weights[0])

font = get_japanese_fonts()[0]
create_wordcloud(count, font)