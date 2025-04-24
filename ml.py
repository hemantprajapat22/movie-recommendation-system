# !/usr/bin/env python
# ## Content Based Recommender System 

import pandas as pd


movies = pd.read_csv('D:\\RecomSys-master\\backend\\ml\\data\\tmdb_5000_movies.csv')
credits = pd.read_csv('D:\\RecomSys-master\\backend\\ml\\data\\tmdb_5000_credits.csv')

movies.head(2)

movies.shape

credits.head()
credits.shape

movies = movies.merge(credits,on='title')
movies.head(2)

movies.shape

# Keeping important columns for recommendation
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head(2)

movies.shape
movies.isnull().sum()

movies.dropna(inplace=True)
movies.isnull().sum()
movies.shape
movies.duplicated().sum()

# handle genres

movies.iloc[0]['genres']

import ast #for converting str to list

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L

movies['genres'] = movies['genres'].apply(convert)

movies.head()

# handle keywords
movies.iloc[0]['keywords']
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# handle cast
movies.iloc[0]['cast']

# only keeping top 3 cast members

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)
movies.head()

# handle crew

movies.iloc[0]['crew']

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

movies.head()

# handle overview (converting to list because we will use count vectorizer)

movies.iloc[0]['overview']
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(4)

movies.iloc[0]['overview']

# removing space
'Bradley Cooper'
'BradleyCooper'

def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(remove_space)
movies.head()

# Concatinate all
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head()

movies.iloc[0]['tags']
movies.head(5)
movies.head()

# Converting list to str
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
movies.head()

# Converting to lower case
movies['tags'] = movies['tags'].apply(lambda x:x.lower())
movies.iloc[0]['tags']


# Removing corrupted and redundant data
index_names = movies[movies['movie_id'] == 113406].index
movies.drop(index_names, inplace=True)
index_names = movies[movies['movie_id'] == 112430].index
movies.drop(index_names, inplace=True)
index_names = movies[movies['movie_id'] == 181940].index
movies.drop(index_names, inplace=True)
movies = movies.drop_duplicates(subset=['movie_id'])

movies.head()

from nltk.stem import PorterStemmer
ps = PorterStemmer()

""" Stemming helps to reduce the dimentionality of the data
 and to focus on the meaning of the word rather than their form. """

def stems(text):
    T = []
    
    for i in text.split():
        T.append(ps.stem(i))
    
    return " ".join(T)


movies['tags'] = movies['tags'].apply(stems)
movies.iloc[0]['tags']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(movies['tags']).toarray()
vector[0]
vector.shape
len(cv.get_feature_names_out())
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)
similarity.shape
movies[movies['title'] == 'The Lego Movie'].index[0]

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(movies.iloc[i[0]].title)
recommend('Spider-Man 2')
movies.head(2)

movies['overview'] = movies['overview'].apply(lambda x: " ".join(x))
movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))
movies['cast'] = movies['cast'].apply(lambda x: " ".join(x))
movies['crew'] = movies['crew'].apply(lambda x: " ".join(x))
movies.head(5)

import pickle
pickle.dump(movies,open('backend/ml/out/cinema.pkl','wb'))
pickle.dump(similarity,open('backend/ml/out/metric.pkl','wb'))