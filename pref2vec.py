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


class Pref2Vec(object):
    '''
    PREF2VEC

    Модель Pref2Vec - модель векторизации пользовательских предпочтений Pref2Vec.
    Модель позволяет построить множество пользовательских пространств эмбеддингов – корпус объектов UES, представленных в виде набора вычисленных центров кластеров взаимодействий. На основе сформированного корпуса открывается возможность исследования пользовательских пространств эмбеддингов, которое позволяет строить рекомендации для конечного пользователя посредством поиска схожих пользовательских предпочтений, то есть схожих пространств. Дальнейшая рекомендация объектов происходит при помощи анализа расширенных эмбеддингов наиболее подобных пользовательских пространств.
    '''


    def __init__(self, depth):
        '''
        Метод инициализации класса, в котором производится определение полей
        `self.depth` и `self.corpus`.

        Аргументы:
        - depth (`int`) -
        '''

        self.depth = depth
        self.corpus = pd.DataFrame(
            columns=['user_id', 'center_read', 'center_shelved',
                     'center_rating_0', 'center_rating_1', 'center_rating_2',
                     'center_rating_3', 'center_rating_4', 'center_rating_5']
        )


    def build_corpus(self, filepath_or_buffer, users_data):
        '''
        Метод, при помощи которого производится построение корпуса
        пользовательских предпочтений.

        Для каждого пользователя рассчитывается пространство эмбеддингов (UES),
        далее полученные центры кластеров взаимодействий сохраняются в корпус
        модели.

        Аргументы:
        - users_data (`pandas.DataFrame`)
        '''

        counter = 0
        for index, row in users_data.iterrows():
            # Расчёт проистранства эмбеддингов
            ues = UES(row)
            # Получение центров кластаров взаимодействий
            centers = ues.retrieve_centers()
            # Добавление данных в корпус
            new_row = {'user_id': ues.user_id}
            for interaction in interactions:
                new_row[f"center_{interaction}"] = self.__get_element_or_nan(
                    array=centers[centers['interaction'] == interaction]['embedding'].values,
                    index=0
                )
            self.corpus.loc[len(self.corpus)] = new_row

            counter += 1
            if (counter % 250 == 0):
                self.__save_corpus(filepath_or_buffer)

        self.corpus = self.corpus.set_index('user_id')
        self.__save_corpus(filepath_or_buffer)


    def __save_corpus(self, filepath_or_buffer):
        '''
        Метод, который сохраняет корпус модели в CSV-файл.

        Аргументы:
        - filepath_or_buffer (`str`) - путь файла, в который произойдёт
            сохранение файла.
        '''

        self.corpus.to_csv(filepath_or_buffer)


    def load_corpus(self, filepath_or_buffer):
        '''
        Метод, позволяющий загрузить корпус модели из CSV-файла.

        Аргументы:
        - filepath_or_buffer (`str`) - путь к используемому CSV-файлу.
        '''

        users_data = pd.read_csv(
            filepath_or_buffer=filepath_or_buffer,
            index_col='user_id'
        )
        # Преобразование строковых значений датасета в значения с плавающей запятой
        for user_id in users_data.index.values:
            for interaction in interactions:
                try:
                    if not (pd.isna(users_data.loc[user_id][f"center_{interaction}"])):
                        center = np.asarray(
                            np.array(re.findall('[-+0-9.e]+', users_data.loc[user_id][f"center_{interaction}"])),
                            dtype=float
                        )
                        users_data.at[user_id, f"center_{interaction}"] = center
                except:
                    continue
        self.corpus = users_data


    def most_similar_users(self, target_user_data, topn=10, show_info=True):
        '''
        Метод, при помощи которого происходит поиск ближайших `topn`
        пользователей к целевому поиску.

        Поиск осуществляется рассчётом косинусных расстоний между центрами
        кластеров категорий искомого пользователя и соответствующими центрами
        пользователей корпуса модели `self.corpus`. Для каждой из категорий
        происходит рассчёт ближайших `self.depth` пользователей. Далее выводятся
        самые близкие пользователи к целевому.

        Аргументы:
        - target_user_data (`pandas.Series`) - серия библиотеки Pandas, в которой
            содержится препроцессированная информация о взаимодействиях
            целевого пользователя с книгами в виде категоризарованного набора
            идентификаторов книг;
        - topn (`int`) - количество ближайших пользователей;
        - show_info (`bool`) - флаг вывода информации о работе метода.

        Возвращает:
        - most_similar_users (`list`) - список с ближашими пользователями.
            Содержит набор объектов `dict`, в которых хранятся идентификатор
            пользователя, категория в которой пользователь оказался близок и
            косинусное расстояние с центром категории целевого пользователя.
        '''

        # Датасет поиска ближайших пользователей
        self.df_most_similar_users = pd.DataFrame(
            columns=['user_id', 'interaction', 'interaction_index', 'similarity']
        )
        # Рассчёт целевого пользовательского пространтва эмбеддингов
        target_ues = UES(target_user_data)
        target_user_id = target_ues.user_id
        target_centers = target_ues.retrieve_centers()

        # Поиск ближайших центров по видам взаимодействий
        for interaction in interactions:
            self.__most_similar_users_by_interaction(
                target_user_id=target_user_id,
                target_center=target_centers[target_centers['interaction'] == interaction]['embedding'].values,
                interaction=interaction
            )
        # Группировка и агрегирование полученный данных
        df_most_similar_users_aggregated = self.df_most_similar_users.groupby('user_id').agg({'interaction': 'count', 'interaction_index': 'mean'})
        df_most_similar_user_top_n = df_most_similar_users_aggregated.sort_values(by=['interaction', 'interaction_index'], ascending=[False, True]).head(topn)
        most_similar_users = list()
        for user_id in df_most_similar_user_top_n.index.values:
            df_current_user = self.df_most_similar_users[self.df_most_similar_users['user_id'] == user_id]
            dict_current_user = {'user_id': user_id, 'appearances': []}
            for index, row in df_current_user.iterrows():
                dict_current_user['appearances'].append((row['interaction'], row['similarity']))
            most_similar_users.append(dict_current_user)

        # Вывод результатов
        if (show_info):
            print(f"Ближайщие пользователи к {target_user_id}:")
            for i in range(len(most_similar_users)):
                print(f"{i+1}) Пользователь №{most_similar_users[i]['user_id']}")
                for appearance in most_similar_users[i]['appearances']:
                    print(f"   {appearance[0]} - {appearance[1]}")

        return most_similar_users


    def __most_similar_users_by_interaction(self, target_user_id, target_center, interaction):
        '''
        Метод, осуществляющий рассчёт косинусных расстоний между искомым центром
        категории и всеми остальными центрами категории корпуса `self.corpus`.

        Расстоние вычисляется при помощи модуля `metrics.pairwise.cosine_similarity`
        библиотеки `sklearn`. Для увеличения скорости вычислений промежуточные
        результаты сохраняются и обрабатываются при помощи библиотеки `pandas`.

        Аргументы:
        - target_user_id (`str`) - идентификатор целевого пользователя;
        - target_center (`list`) - массив, содержащий целевой центр кластера
            текущего взаимодействия;
        - interaction ('str') - рассматриваемое взаимодействие.
        '''

        # Если у пользователя нет эмбеддингов текущего взаимодействия, ближайщих пользователей нет
        if (target_center.size == 0):
            return

        # Инициализация датасета, в котором будут рассчитываться расстония
        df_similarity = pd.DataFrame(data={'user_id': self.corpus.index.values, 'center': self.corpus[f"center_{interaction}"]})
        # Фильтрация элементов, в которых нет центра рассматриваемой категории
        df_similarity = df_similarity[df_similarity['center'].notna()]
        # Вычисление расстояний между искомым центром и всеми остальными
        target_center = [target_center[0]] if (target_center.size > 1) else list(target_center)
        similarities = cosine_similarity(target_center, list(df_similarity['center']))
        df_similarity['similarity'] = np.reshape(similarities, df_similarity['center'].size)
        # Обработка случая, в котором найденное рассторие - расстояние с искомым центром
        if (df_similarity['similarity'].idxmax() == target_user_id):
            df_similarity = df_similarity.drop(df_similarity['similarity'].idxmax())

        # Сохранение ближайших пользователей в датасет self.df_most_similar_users
        df_most_similar_users = pd.DataFrame(
            data={
                'user_id': df_similarity['similarity'].nlargest(self.depth).index.values,
                'interaction': [interaction] * self.depth,
                'interaction_index': [i+1 for i in range(self.depth)],
                'similarity': df_similarity['similarity'].nlargest(self.depth).values
            }
        )
        self.df_most_similar_users = pd.concat([self.df_most_similar_users, df_most_similar_users])


    def recommend_for_user(self, target_user_data, topn=50, show_info=True):
        '''
        Метод, при помощи которого происходит рассчёт `topn` рекомендаций для
        целевого пользователя.

        Сначала рассчитываются 10 ближайших пользователей к целевому. Далее
        рассматриваются все объекты ближащий пользователей на предмет
        косинусного расстояния к объектам целевого пользоваля - происходит
        рассчёт расстояний между рассматриваемым объектом и всеми целевыми,
        наименьшее расстояние сохраняется. После происходит сортировка объектов
        по полученным расстояниям и вывод `topn` ближайших.

        Аргументы:
        - target_user_data (`pandas.Series`) - серия библиотеки Pandas, в которой
            содержится препроцессированная информация о взаимодействиях
            целевого пользователя с книгами в виде категоризарованного набора
            идентификаторов книг;
        - topn (`int`) - количество ближайших пользователей;
        - show_info (`bool`) - флаг вывода информации о работе метода.

        Возвращает:
        - recommendations (`list`) - список с полученными рекомендациями.
            Содержит пары с идентификатором рекомендуемого объекта и ближайщим
            расстоянием.
        '''

        # Рассчёт целевого пользовательского пространтва эмбеддингов
        ues_target = UES(target_user_data)
        ues_target.df = ues_target.df[ues_target.df['interaction'].isin(target_interactions)]
        # Датасет, в котором будут рассчитываться косинусные расстояния
        self.df_embeggings_target = pd.DataFrame(
            data={'embedding': ues_target.df['embedding'].values},
            index=[re.findall('[0-9][0-9]+', index)[0] for index in ues_target.df.index.values]
        )

        # Поиск ближайших пользователей
        most_similar_users = self.most_similar_users(target_user_data, show_info=False)
        most_similar_users_ids = [x['user_id'] for x in most_similar_users]
        # Датасет с расстояниями всех объектов с искомыми
        self.df_recommendations = pd.DataFrame(columns=['book_id', 'similarity'])
        # Прогон объектов всех ближайших пользователей
        for user_id in most_similar_users_ids:
            ues_near = UES(df_user_data.loc[user_id])
            ues_near.df = ues_near.df[ues_near.df['interaction'].isin(target_interactions)]
            # Вычисление ближайшего расстония для каждого объекта с целевыми объектами
            for index, embedding in ues_near.df['embedding'].items():
                nearest_similarity = self.__get_nearest_similarity(
                    embedding=embedding,
                    embedding_id=re.findall('[0-9][0-9]+', index)[0]
                )
                self.df_recommendations.loc[len(self.df_recommendations.index)] = {
                    'book_id': re.findall('[0-9][0-9]+', index)[0],
                    'similarity': nearest_similarity
                }

        # Сортировка объектов и вывод `topn` ближайших
        self.df_recommendations = self.df_recommendations.drop_duplicates('book_id', keep='first')
        self.df_recommendations = self.df_recommendations.sort_values(by='similarity', ascending=False).head(topn)
        recommendations = []
        for index, row in self.df_recommendations.iterrows():
            recommendations.append((row['book_id'], row['similarity']))

        # Вывод результатов
        if (show_info):
            print(f"Рекомендации для пользователя {ues_target.user_id}:")
            counter = 1
            for index, row in self.df_recommendations.iterrows():
                print(f"{counter}) Книга #{row['book_id']} - {row['similarity']}")
                counter += 1

        return recommendations


    def __get_nearest_similarity(self, embedding, embedding_id):
        '''
        Метод, который позволяет найти ближайшее косинусное расстояние к одному
        из целевых объектов датасета `self.df_embeggings_target`.

        Аргументы:
        - embedding (`list`) - эмбеддинг расссматриваемого объекта;
        - embedding_id (`str`) - идентификатор расссматриваемого объекта.
        '''

        # Рассчёт растояний между рассматриваемым объектом и целевыми
        similarities = cosine_similarity([embedding], list(self.df_embeggings_target['embedding']))
        self.df_embeggings_target['similarity'] = np.reshape(similarities, self.df_embeggings_target['embedding'].size)

        # Обработка случая, в котором найденное рассторие - расстояние с искомым объектом
        if (self.df_embeggings_target['similarity'].idxmax() == embedding_id):
            self.df_embeggings_target.drop(self.df_embeggings_target['similarity'].idxmax())

        return self.df_embeggings_target['similarity'].max()


    def __get_element_or_nan(self, array, index):
        '''
        Метод, который возвращает элемент массива, если его адрес находится
        в пределах массива. В ином случае возвращается элемент `np.nan`.

        Аргументы:
        - array (`array-like`) - массив, из которого необходимо получить элемент;
        - index (`int`) - индекс получаемого элемента.
        '''

        try:
            return array[index]
        except IndexError:
            return np.nan
