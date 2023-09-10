import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake

import numpy as np
# from gensim.summarization import keywords
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def get_all_unique_data(col_name, data_path='server/data/all_data.csv'):
    table = pd.read_csv(data_path)

    return table[col_name].unique()

def get_all_stat(data_path='server/data/labeled_data.csv'):
    table = pd.read_csv(data_path)


    return {'cluster_count': len(table['answers_cluster'].unique()), 'people_count': table['answers_count'].sum(), 'max_in_cluster':   table['answers_count'].max()}

def get_by_query(query, data_path='server/data/labeled_data.csv'):
    table = pd.read_csv(data_path)
    query_data = table[table['query'] == query]

    return query_data.to_json()

def get_clusters(logs_q, data_path='server/data/all_data_sifted_2.csv'):
    qa_dict = {}
    table = pd.read_csv(data_path)
    query_data = table[table['query'] == logs_q].reset_index()

    
    query = query_data['query']
    
    answers_text = query_data['answers_text']

    
    for index in range(len(query)):
        print(index)
        if query[index] not in qa_dict.keys():
            qa_dict[query[index]] = []    
        qa_dict[query[index]].append(answers_text[index])
    
    qa_dict_token = {}

    for query in qa_dict.keys():
        for answer in qa_dict[query]:
            if query not in qa_dict_token.keys():
                qa_dict_token[query] = []
            qa_dict_token[query].append(nltk.tokenize.word_tokenize(answer))

    stop_words = nltk.corpus.stopwords.words('russian')

    stop_words.append('?')
    stop_words.append('!')
    stop_words.append(',')
    stop_words.append('.')
    stop_words.append('-')
    stop_words.append(',')
    stop_words.append('«')
    stop_words.append('»')
    stop_words.append('-')
    stop_words.append('(')
    stop_words.append(')')
    stop_words.append('>')
    stop_words.append('<')

    stop_words.extend(nltk.corpus.stopwords.words('english'))

    stop_words.append('@')
    stop_words.append('#')
    stop_words.append('http')
    stop_words.append(':')

    qa_dict_non_stop = {}

    for query in qa_dict_token.keys():
        one_query_lst = []
        for one_answer in qa_dict_token[query]:
            one_answer_non_stop = [w.lower() for w in one_answer if w not in stop_words]
            # one_processed_sentence = list(set(one_processed_sentence))
            one_query_lst.append(one_answer_non_stop)
        qa_dict_non_stop[query] = one_query_lst
    
    qa_dict_detoken = {}

    for query in qa_dict_non_stop.keys():
        one_query_lst = []
        for one_answer in qa_dict_non_stop[query]:
            # print(type(one_sentence))
            if len(one_answer) == 0:
                one_query_lst.append('')
                continue
            answer = ''
            answer += one_answer[0]
            for index in range(1, len(one_answer)):
                answer += ' '
                answer += one_answer[index]
            one_query_lst.append(answer)
        qa_dict_detoken[query] = one_query_lst

    tfidf_dict = {}

    for one_query in qa_dict_detoken.keys():        
        # one_tfidf = []

        if len(qa_dict_detoken[one_query]) == 0:
            tfidf_dict[one_query] = one_tfidf
            continue
    
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000, max_df = 1.0, min_df = 0.01, use_idf = True, ngram_range=(1,3))
        try:
            X = vectorizer.fit_transform(qa_dict_detoken[one_query])
            print(X.shape) # check shape of the document-term matrix
            # one_tfidf.append(X)
            tfidf_dict[one_query] = X
            # terms = vectorizer.get_feature_names_out()
        except:
            print(qa_dict_detoken[one_query])
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold = 0.5)
    rake_nltk_var = Rake()

    clusters_dict = {}

    for each_key in tfidf_dict.keys():
        try:
            # print(type(tfidf_dict[each_key]))
            clustering.fit(tfidf_dict[each_key].toarray())
            cluster_labels = clustering.labels_
            answers = qa_dict_detoken[each_key]
            cluster_arr = []

            for cluster_label in range(max(cluster_labels)):
                cluster_texts = [answers[i] for i in cluster_labels if i == cluster_label]

                cluster_name = cluster_texts[0]
                power = len(cluster_texts)

                # cluster_text = ' '.join(cluster_texts)
                # cluster_keywords = keywords(cluster_text, words=3).split('\n')
                cluster_arr.append({'name': cluster_name, 'power': power})



            clusters_dict[each_key] = cluster_arr
        except:
            clusters_dict[each_key] = None
    return clusters_dict

if __name__ == "__main__":
    print(get_all_unique_data('query'))
    print(get_all_stat())
    print(get_by_query('Вопрос 8. Какая поддержка нам нужна от руководителей отрасли и дивизиона?'))
    print(get_clusters('Что вы сможете использовать в своей работе?'))
