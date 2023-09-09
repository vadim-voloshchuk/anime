import pandas as pd

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


if __name__ == "__main__":
    print(get_all_unique_data('query'))
    print(get_all_stat())
    print(get_by_query('Вопрос 8. Какая поддержка нам нужна от руководителей отрасли и дивизиона?'))
