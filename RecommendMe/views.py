# i have created this file
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
def index(request):
    return render(request,'index.html')

def about_us(request):
    return render(request,'about_us.html')

def option(request):
    x=request.GET.get('search','default')
    df1=pd.read_csv("RecommendMe\mlens\credits.csv")
    df2=pd.read_csv("RecommendMe\mlens\movies.csv")
    df1.columns = ['id','tittle','cast','crew']
    df2= df2.merge(df1,on='id')
    list1=[]
    for i in range(len(df2)):
        list1.append(df2['title'][i].lower())
    list2=[]
    ls5=[]
    for i in range(len(list1)):
        if str(x).lower() in list1[i]:
            list2.append(list1[i])
    for i in range(len(list2)):
        for j in range(len(list2)):
            if list2[i]==df2['title'][j].lower():
                ls5.append(df2.poster[j])

    #l=df2.poster.tolist()
    #l=l[:100]
    #l1=df2.title.tolist()
    #l1=l1[:100]
    mylist = zip(ls5,list2)
    param={'mlist':mylist}        
    return render(request,'option.html',param)


def recommend(request):
    name=request.GET.get('search','default')
    name=str(name)
    df1=pd.read_csv("RecommendMe\mlens\credits.csv")
    df2=pd.read_csv("RecommendMe\mlens\movies.csv")
    df1.columns = ['id','tittle','cast','crew']
    df2= df2.merge(df1,on='id')
    df2['title']=df2['title'].str.lower()
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words='english')

    df2['overview'] = df2['overview'].fillna('')

    #Construct the required TF-IDF matrix by fitting and transforming the data
    matrix = tfidf.fit_transform(df2['overview'])
    from sklearn.metrics.pairwise import linear_kernel

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(matrix, matrix)
    indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
    def get_recommendations(title, cosine_sim=cosine_sim):
        # Get the index of the movie that matches the title
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:19]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return df2['title'].iloc[movie_indices]
    ls=get_recommendations(name.lower()).tolist()
    ls2=[]#overview
    ls3=[]#release date
    ls4=[]#title
    ls5=[]#poster
    for i in range(len(ls)):
        for j in range(len(df2)):
            if df2['title'][j]==ls[i]:
                ls2.append(df2['overview'][j])
                ls3.append(df2['release_date'][j])
                ls4.append(df2['original_title'][j])
                ls5.append(df2['poster'][j])

    length=len(ls)
    mylist = zip(ls4,ls5)
    param={'list':mylist,'length':length}
    return render(request,'recommend.html',param)