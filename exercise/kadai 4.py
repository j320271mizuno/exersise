#%%
from chapter01 import get_string_from_file
from chapter02 import get_words_from_file
from chapter03 import (add_to_corpus, bows_to_cfs, get_bows,
                       get_tfidfmodel_and_weights, get_weights, get_words,
                       load_aozora_corpus, translate_bows)
from gensim import corpora, models

#%%
texts = ['おまえのものはおれのもの、おれのものもおれのもの。','ジャイアンとのび太とスネ夫']
words = [get_words(text, keep_pos=['名詞']) for text in texts]
dic = corpora.Dictionary(words)
for i in range(len(dic)):
    print('dic[%d] = %s' % (i, dic[i]))
bows = [dic.doc2bow(w) for w in words]
tfidf = models.TfidfModel(bows)
weights = tfidf[bows[0]]
weights = [(i, round(j, 4)) for i, j in weights]
print('weights =', weights)

#%%
from gensim.similarities import MatrixSimilarity


def vsm_search(texts, query):
    tfidf_model, dic, text_weights = get_tfidfmodel_and_weights(texts)
    index = MatrixSimilarity(text_weights, num_features=len(dic))
    query_bows = get_bows([query], dic)
    query_weights = get_weights(query_bows, dic, tfidf_model)
    sims = index[query_weights[0]]
    return sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

def get_list_from_file(file_name):
    with open(file_name, 'r', encoding='UTF-8') as f:
        return f.read().split()

texts = [get_string_from_file('data/ch04/%d.txt' % i) for i in range(12)]
titles = get_list_from_file('data/ch04/book_titles.txt')
query = '数学'
result = vsm_search(texts, query)
for x in range(len(result)):
    print('%s %.4f' % (titles[result[x][0]], result[x][1]))