import numpy as np
import pandas as pd
import re

from gensim.parsing.preprocessing import preprocess_string
from gensim.models.doc2vec import Doc2Vec

from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud


READ     = 'read'
SHELVED  = 'shelved'
RATING_0 = 'rating_0'
RATING_1 = 'rating_1'
RATING_2 = 'rating_2'
RATING_3 = 'rating_3'
RATING_4 = 'rating_4'
RATING_5 = 'rating_5'
interactions = [SHELVED, READ, RATING_0, RATING_1, RATING_2, RATING_3, RATING_4, RATING_5]

left_boundary = -1
right_boundary = 1
step = (right_boundary - left_boundary) / (len(interactions) - 1)

VALUE_READ     = left_boundary + 0 * step
VALUE_SHELVED  = left_boundary + 1 * step
VALUE_RATING_0 = left_boundary + 2 * step
VALUE_RATING_1 = left_boundary + 3 * step
VALUE_RATING_2 = left_boundary + 4 * step
VALUE_RATING_3 = left_boundary + 5 * step
VALUE_RATING_4 = left_boundary + 6 * step
VALUE_RATING_5 = left_boundary + 7 * step
VALUE_DEFAULT  = np.nan

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


class UES(object):
    '''
    USER EMBEDDING SPACE

    Класс, который описывает пространство эмбеддингов конкретного пользователя.
    Экземпляр содержит в себе набор эмбеддингов, учитывающих, как и объект,
    так и вид взаимодействия с ним, и набор средств анализа рассчитанного
    пространства.
    '''


    def __init__(self, user_data):
        '''
        Метод инициализации класса, в котором производится обращение к методам,
        формирующим собственно пространство эмбеддингов.

        Аргументы:
        - user_data  (`pandas.Series`) - серия библиотеки Pandas, в которой
            содержится препроцессированная информация о взаимодействиях
            пользователя с книгами в виде категоризарованного набора
            идентификаторов книг.
        '''

        self.user_id = user_data.name
        # Датасет, в котором хранится все рассчитанные данные
        self.df = pd.DataFrame(columns=['id', 'embedding', 'interaction', 'type', 'tokens'])
        # Формирование пространства эмбеддингов
        self.__construct_embedding_space(user_data)
        self.__compute_centers()


    def __construct_embedding_space(self, user_data):
        '''
        Метод, при помощи которого происходит расчёт эмбеддингов: формирование
        эмбеддинга с помощью модели Doc2Vec и его последующая корректировка.

        Аргументы:
        - user_data  (`pandas.Series`) - серия библиотеки Pandas, в которой
            содержится препроцессированная информация о взаимодействиях
            пользователя с книгами в виде категоризарованного набора
            идентификаторов книг.
        '''

        for interaction in interactions:
            # Идентификаторы книг текущего вида взаимодейсвия
            ids = re.findall('[0-9]+', user_data[interaction])
            for id in ids:
                # Если имеется аннотация рассматриваемой, происходит формирование эмбеддинга
                try:
                    # Формирование изначального эмбеддинга
                    text_tokens = preprocess_string(df_corpus_annotations.loc[int(id)]['annotation'])
                    text_embedding = model_d2v.infer_vector(text_tokens)
                    # Добавление измерения, характеризующего вид взаимодействия
                    interaction_embedding = np.append(text_embedding, self.__get_interaction_value(interaction))
                    # Запись полученного эмбеддинга в датасет пространства
                    new_row = {'id': f"{interaction}_{id}",
                               'embedding': interaction_embedding,
                               'interaction': interaction,
                               'type': 'object',
                               'tokens': ' '.join(text_tokens)}
                    self.df.loc[len(self.df)] = new_row
                except:
                    continue
        self.df = self.df.set_index('id')


    def __get_interaction_value(self, interaction):
        '''
        Метод, который предоставляет значение дополнительной компоненты
        эмбеддинга в зависимости от вида взаимодействия пользователя с книгой.

        Аргументы:
        - interaction  (`str`) - вид взаимодействия пользователя с книгой.
        '''

        match interaction:
            case 'shelved':
                return VALUE_SHELVED
            case 'read':
                return VALUE_READ
            case 'rating_0':
                return VALUE_RATING_0
            case 'rating_1':
                return VALUE_RATING_1
            case 'rating_2':
                return VALUE_RATING_2
            case 'rating_3':
                return VALUE_RATING_3
            case 'rating_4':
                return VALUE_RATING_4
            case 'rating_5':
                return VALUE_RATING_5
            case _:
                return VALUE_DEFAULT


    def __compute_centers(self):
        '''
        Метод, позволяющий получить центры (медоиды) кластеров взаимодействий
        при помощи алгоритма K-Medoids.

        Все вычисленные центры кластеров помечаются в датасете `self.df`.
        '''

        for interaction in interactions:
            # Элементы кластера текущего вида взаимодействия
            interaction_data = self.df[self.df['interaction'] == interaction]
            embeddings = interaction_data['embedding'].values.tolist()
            # Если текущий кластер не является пустым, происходит поиск его центра
            if not (len(embeddings) == 0):
                # Поиск центра кластера при помощи алгоритма K-Means
                kmedoids = KMedoids(n_clusters=1)
                kmedoids.fit(embeddings)
                center = kmedoids.cluster_centers_[0]
                # Отметка центра в датасете пространства
                for index, row in interaction_data.iterrows():
                    if (np.array_equal(row['embedding'], center)):
                        self.df.at[index, 'type'] = 'center'
                        break


    def retrieve_centers(self):
        '''
        Метод, возвращающий раннее вычисленные центры кластеров взаимодействий
        в виде фрагмента датасета.

        Возвращает:
        - df (`pandas.DataFrame`) - датасет с центрами кластеров.
        '''

        return self.df[self.df['type'] == 'center']


    def show_tsne(self, is_text_embeddings=False):
        '''
        Метод, при помощи которого происходит визуализация пространства
        эмбеддингов пользователя сниженной размерности при помощи
        алгоритма T-SNE.

        Аргументы:
        - is_text_embeddings (`bool`) - если аргумент равен `True`, то происходит
            визуализация исходного пространства эмбеддингов, иначе -
            пространства эмбеддингов с учётом вида взаимодействия.
        '''

        # Размерность эмбеддингов очределяет учёт взаимодействий
        embedding_size = VECTOR_SIZE if (is_text_embeddings) else VECTOR_SIZE + 1
        # Формирование итоговых эмбеддингов
        embeddings = np.zeros((len(self.df['embedding']), embedding_size))
        for i in range(len(self.df['embedding'])):
            embeddings[i,:] = np.array(
                self.df.iloc[i]['embedding'][:embedding_size]
            ).reshape((1, embedding_size))
        # Уменьшение размерности при помощи модели T-SNE
        perplexity = len(embeddings) - 2 if (len(embeddings) < 50) else 30
        model_tsne = TSNE(
            perplexity=perplexity, n_components=3, init='pca'
        )

        # Сохрание полученных значений в датасет пространства
        tsne_embeddings = model_tsne.fit_transform(embeddings)
        tsne_x = list(map(lambda x: x[0], tsne_embeddings))
        tsne_y = list(map(lambda x: x[1], tsne_embeddings))
        tsne_z = list(map(lambda x: x[2], tsne_embeddings))
        if (is_text_embeddings):
            self.df['tsne_x_text'] = tsne_x
            self.df['tsne_y_text'] = tsne_y
            self.df['tsne_z_text'] = tsne_z
        else:
            self.df['tsne_x_interaction'] = tsne_x
            self.df['tsne_y_interaction'] = tsne_y
            self.df['tsne_z_interaction'] = tsne_z

        # Визуализация полученных результатов
        if (is_text_embeddings):
            title = 'Визуализация UES при помощи метода T-SNE<br>(без учёта взаимодействий)'
            x = 'tsne_x_text'
            y = 'tsne_y_text'
            z = 'tsne_z_text'
        else:
            title = 'Визуализация UES при помощи метода T-SNE<br>(с учётом взаимодействий)'
            x = 'tsne_x_interaction'
            y = 'tsne_y_interaction'
            z = 'tsne_z_interaction'
        # Построение 3-ёх мерной диаграммы рассеяния
        fig_1 = px.scatter_3d(
            self.df, x=x, y=y, z=z, color='interaction', symbol='type',
            labels={'interaction': 'Вид взаимодействия', 'type': 'Тип объекта'}
        )
        fig_1.update_layout(
            height=470, width=800, title=dict(text=title, font=dict(size=18)),
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )
        fig_1.show()
        # Построение матрицы рассеяния
        fig_2 = px.scatter_matrix(
            self.df, dimensions=[x, y, z], color='interaction',
            labels={'interaction': "Вид взаимодействия", x: 'X', y: 'Y', z: 'Z'}
        )
        fig_2.update_layout(
            height=470, width=800,
            title=dict(text='Матрица диаграмм рассеяния', font=dict(size=18))
          )
        fig_2.show()


    def show_wordclouds(self):
        '''
        Метод, визуализирующий кластеры взаимодействий посредством
        формирования облака слов.
        '''

        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Shelved', 'Read', 'Rating - 0', 'Rating - 1',
                            'Rating - 2', 'Rating - 3', 'Rating - 4', 'Rating - 5')
        )
        index = 0
        for interaction in interactions:
            row_index = int(index / 3 + 1)
            col_index = int(index % 3 + 1)
            if not (len(self.df[self.df['interaction'] == interaction]) == 0):
                # Если текущий кластер не пустой, происходит построение облака слов
                text = ' '.join(self.df[self.df['interaction'] == interaction]['tokens'])
                wordcloud = WordCloud(
                    width=500, height=500, min_font_size=10, background_color="white"
                ).generate(text)
                fig.add_trace(px.imshow(wordcloud).data[0], row=row_index, col=col_index)
                fig.update_xaxes(visible=False, row=row_index, col=col_index)
                fig.update_yaxes(visible=False, row=row_index, col=col_index)
            else:
                # Если текущий кластер пустой, происходит вывод пустого облака слов
                wordcloud = WordCloud(
                    width=500, height=500, min_font_size=10, background_color="white"
                ).generate('Empty')
                fig.add_trace(px.imshow(wordcloud).data[0], row=row_index, col=col_index)
                fig.update_xaxes(visible=False, row=row_index, col=col_index)
                fig.update_yaxes(visible=False, row=row_index, col=col_index)
            index += 1
        fig.update_layout(height=900, width=850,
                          title=dict(text='Облака слов в зависимости от реакции пользователя',
                                     font=dict(size=18)))
        fig.show()
        