import numpy as np
import pandas as pd

import re
import collections
import datetime

from gensim.parsing.preprocessing import preprocess_string
from gensim.models.doc2vec import Doc2Vec

from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud


target_interactions = ['read', 'rating_4', 'rating_5']

df_user_data = pd.read_csv(
    filepath_or_buffer='/content/drive/MyDrive/ВКР/children/children_user_data.csv',
    index_col='user_id'
)
df_corpus_annotations = pd.read_csv(
    filepath_or_buffer='/content/drive/MyDrive/ВКР/children/children_corpus_annotations.csv',
    index_col='book_id'
)

filename_d2v = '/content/drive/MyDrive/ВКР/children/children_model_d2v.d2v'
model_d2v = Doc2Vec.load(filename_d2v)
VECTOR_SIZE = model_d2v.vector_size


class Evaluator:
    '''
    RECOMMENDER SYSTEM EVALUATOR

    Класс Evaluator позволяет провести тестирование работы рекомендательной
    системы на предмет качества формирования списков рекомендаций. Также этот
    класс позволяет произвести продготовку тестовых данных (данных о
    взаимодействиях пользователей с различными объектами).
    '''


    def __init__(self, model_p2v, test_percent):
        '''
        Метод инициализации класса, в котором производится инициализация полей
        self.model_p2v (модель Pref2Vec) и self.test_percent (процент разбиения
        данных на выборки X и Y).

        Аргументы:
        - model_p2v ('Pref2Vec') - модель Pref2Vec, работа которой будет
            тестироваться;
        - test_percent ('int') - процент разбиения данных на выборки X и Y.
        '''

        self.model_p2v = model_p2v
        self.test_percent = test_percent


    def test(self, users_data, filepath_or_buffer, k=20):
        '''
        Метод, который производит тестирование модели `model_p2v` на
        пользовательских данных `users_data`.

        Используются метрики NDCD@k, Recall@k и Precision@k. Результаты
        тестирования сохраняются в поле `self.df_testing`. Кроме того, имеется
        возможность загрузки результатов в CSV-файл.

        Аргументы:
        - users_data ('pandas.DataFrame') - пользовательские данные,
            используемые при тестировании рекомендательной системы;
        - filepath_or_buffer ('str') - путь к CSV-файлу, в который будут
            сохранены результаты тестирования;
        - k ('int') - параметр 'k', используемый в метриках NDCD@k, Recall@k и
            Precision@k.
        '''

        self.df_testing = pd.DataFrame(columns=['user_id', f"ndcg@{k}", f"recall@{k}", f"precision@{k}"])
        self.k = k

        counter = 1
        for index, row in users_data.iterrows():
            try:
                # Разделение пользовательских данных
                x, y, y_true = self.__sample_preparation(index, row)
                # Получение рекомендаций для текущего пользователя
                recommendations = model_p2v.recommend_for_user(target_user_data=x.iloc[0], topn=k, show_info=False)
                y_pred = [(lambda x: x[0])(x) for x in recommendations]
                # Тестирование и запись результатов в self.df_testing
                ndcg = self.ndcg_at_k(r=[x in y_true for x in y_pred])
                recall = self.recall_at_k(y_true, y_pred)
                precision = self.precision_at_k(y_true, y_pred)
                print(f"{counter}) User {index}:")
                print(f"    ndcg@{k}      = {ndcg}")
                print(f"    recall@{k}    = {recall}")
                print(f"    precision@{k} = {precision}")
                self.df_testing.loc[len(self.df_testing)] = {
                    'user_id': index,
                    f"ndcg@{k}": ndcg,
                    f"recall@{k}": recall,
                    f"precision@{k}": precision
                }
                # Периодическая запись CSV-файла с результатами тестирования
                counter += 1
                if (counter % 5 == 0):
                    self.df_testing.to_csv(filepath_or_buffer)
            except:
                continue

        # Запись CSV-файла с результатами тестирования
        self.df_testing.to_csv(filepath_or_buffer)


    def __sample_preparation(self, user_id, user_data):
        '''
        Метод разбиения тестовых данных на выборки X и Y и ранжирования выборки Y.

        Аргументы:
        - user_id ('str') - идентификатор пользователя, чьи данные будут
            подвержены разбиению;
        - user_data ('pandas.Series') - собственно данные, которые будут
            подвержены разбиению.

        Возвращаются:
        - self.x ('list') - формируемая выборка X;
        - self.y ('list') - формируемая выборка Y;
        - self.y_true ('list') - ранжированная выборка Y;
        '''

        self.x = pd.DataFrame(columns=['user_id', 'read', 'shelved', 'rating_0', 'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5'])
        self.y = pd.DataFrame(columns=['user_id', 'read', 'shelved', 'rating_0', 'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5'])
        row_x = {'user_id': user_id}
        row_y = {'user_id': user_id}

        # Разделение пользовательских данных на выборки X и Y
        for interaction in interactions:
            ids = re.findall('[0-9]+', user_data[interaction])
            x_size = int(len(ids) * (100 - self.test_percent) / 100)
            if (interaction in target_interactions):
                row_x[interaction] = str(ids[:x_size])
                row_y[interaction] = str(ids[x_size:])
            else:
                row_x[interaction] = str(ids)
                row_y[interaction] = str([])
        self.x.loc[0] = row_x
        self.y.loc[0] = row_y

        # Ранжирование выборки Y
        self.y_true = self.__get_y_true(self.x.iloc[0], self.y.iloc[0])

        return (self.x, self.y, self.y_true)


    def __get_y_true(self, x, y):
        '''
        Метод, осуществляющий ранжирование рассматриваемой выборки Y.

        Вычисляется косинусное сходство каждого объекта выборки Y с каждым
        эмбеддингом выборки X, сохранив максимальные значения сходства. Далее
        элементы выборки Y ранжируются по вычисленным сходствам и формируется
        список `y_true`.

        Аргументы:
        - x ('pandas.Series') - выборка X;
        - y ('pandas.Series') - выборка Y.

        Возвращается:
        - self.df_embeggings_y.index.values - ('list') - ранжированный список
            данных выборки Y.
        '''

        # Формирование целевого пользовательского пространства X
        ues_x = UES(x)
        ues_x.df = ues_x.df[ues_x.df['interaction'].isin(target_interactions)]
        self.df_embeggings_x = pd.DataFrame(
            data={'embedding': ues_x.df['embedding'].values},
            index=re.findall('[0-9][0-9]+', str(ues_x.df.index.values))
        )
        # Формирование исследуемого пользовательского пространства Y
        ues_y = UES(y)
        ues_y.df = ues_y.df[ues_y.df['interaction'].isin(target_interactions)]
        self.df_embeggings_y = pd.DataFrame(
            data={'embedding': ues_y.df['embedding'].values, 'distance': [0] * len(ues_y.df['embedding'].values)},
            index=re.findall('[0-9][0-9]+', str(ues_y.df.index.values))
        )
        # Поиск наиболее схожих элементов в пространстве Y с элементами пространства X
        for index, row in self.df_embeggings_y.iterrows():
            self.df_embeggings_y.loc[index, 'distance'] = self.__get_nearest_distance(
                embedding=row['embedding'],
                embedding_id=index
            )
        self.df_embeggings_y = self.df_embeggings_y.sort_values(by='distance', ascending=False).head(self.k)

        return self.df_embeggings_y.index.values


    def __get_nearest_distance(self, embedding, embedding_id):
        '''
        Метод, при помощи которого вычисляется косинусное сходство эмбеддинга
        объекта выборки Y с каждым эмбеддингом выборки X. Выводит максимальное
        значение сходства.

        Аргументы:
        - embedding ('list') - эмбеддинг рассматриваемого объекта;
        - embedding_id ('str') - индентификатор рассматриваемого объекта.

        Возвращается:
        - self.df_embeggings_x['distance'].max() ('float') - максимальное
            значение сходства.
        '''

        # Рассчёт растояний между рассматриваемым объектом и целевыми
        distances = cosine_similarity([embedding], list(self.df_embeggings_x['embedding']))
        self.df_embeggings_x['distance'] = np.reshape(distances, self.df_embeggings_x['embedding'].size)

        # Обработка случая, в котором найденное рассторие - расстояние с искомым объектом
        if (self.df_embeggings_x['distance'].idxmax() == embedding_id):
            self.df_embeggings_x.drop(self.df_embeggings_x['distance'].idxmax())

        return self.df_embeggings_x['distance'].max()


    def precision_at_k(self, y_true, y_pred):
        '''
        Метод, позволяющий получить значение метрики Precision@k.

        Аргументы:
        - y_true ('list') - эталонный список рекомендаций;
        - y_pred ('list') - список рекомендаций, формируемый
            рекомендательной системой;

        Возвращает:
        - precision ('float') - значение метрики Precision@k
        '''

        set_y_true = set(y_true)
        set_y_pred = set(y_pred[:self.k])

        precision = len(set_y_true & set_y_pred) / float(self.k)

        return precision


    def recall_at_k(self, y_true, y_pred):
        '''
        Метод, позволяющий получить значение метрики Recall@k.

        Аргументы:
        - y_true ('list') - эталонный список рекомендаций;
        - y_pred ('list') - список рекомендаций, формируемый
            рекомендательной системой;

        Возвращает:
        - recall ('float') - значение метрики Recall@k
        '''

        set_y_true = set(y_true)
        set_y_pred = set(y_pred[:self.k])

        recall = len(set_y_true & set_y_pred) / float(len(set_y_true))

        return recall


    def dcg_at_k(self, r, method=0):
        '''
        Метод, позволяющий получить значение метрики DCG@k.

        Аргументы:
        - y_true ('list') - эталонный список рекомендаций;
        - y_pred ('list') - список рекомендаций, формируемый
            рекомендательной системой;

        Возвращает:
        - dcg ('float') - значение метрики DCG@k
        '''

        r = np.asfarray(r)[:self.k]

        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')

        return 0.


    def ndcg_at_k(self, r, method=0):
        '''
        Метод, позволяющий получить значение метрики NDCG@k.

        Аргументы:
        - y_true ('list') - эталонный список рекомендаций;
        - y_pred ('list') - список рекомендаций, формируемый
            рекомендательной системой;

        Возвращает:
        - ndcg ('float') - значение метрики NDCG@k
        '''

        dcg_max = self.dcg_at_k(sorted(r, reverse=True), method)

        if not dcg_max:
            return 0.

        return self.dcg_at
