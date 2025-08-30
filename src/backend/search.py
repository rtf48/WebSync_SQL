
import numpy as np
import copy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from backend.scratch import filter_fanfics
import backend.sql_setup as sql_setup
from backend.models import Novels, Cossims, Fanfics
import time


"""========================== Backend functions: ============================="""

sql_setup.setup()


def webnovel_to_top_fics(webnovel_title, num_fics, popularity_weight, user_input_tags):
    """
    Called when the user clicks "Show Recommendations"
    inputs: 
    webnovel_title --> the title of the user queried webnovel
    num_fics: the number of results we output <50
    outputs:
    the top 10 fanfiction information. Can include: 
    - fanfic_id
    - fanfic_titles
    - descriptions
    - etc.
    """

    #put together sorted fanfics tuple list and get top n
    sorted_fanfics_tuplst = Cossims.query.filter_by(webnovel_title=webnovel_title).first().similar_fic_titles
    top_n = copy.deepcopy(sorted_fanfics_tuplst[:num_fics])
    #calculate popularities and sort
    start_time = time.time()
    relevant_fics = [Fanfics.query.filter_by(idx=fic_tuple[1]).first() for fic_tuple in top_n]
    max_pop = np.max([fic.popularity for fic in relevant_fics]) / 10
    if max_pop > 0:
        for i in range(len(top_n)):
            fic_tuple = top_n[i]
            fic_tuple[0] = relevant_fics[i].popularity / max_pop * popularity_weight + fic_tuple[0] * (1 - popularity_weight)
    else:
        for i in range(len(top_n)):
            fic_tuple = top_n[i]
            fic_tuple[0] = fic_tuple[0] * (1 - popularity_weight)
    sorted_pairs = sorted(zip(top_n, relevant_fics), key=lambda x: x[0], reverse=True)[:num_fics]
    top_n, relevant_fics = map(list, zip(*sorted_pairs))

    #populate results
    top_n_fanfics = []
    count = 0
    for i in range(len(top_n)):
        fic = relevant_fics[i]
        info_dict = {}
        info_dict["fanfic_id"] = fic.id                 # get fanfic id
        info_dict["description"] = fic.description      # get description
        info_dict["title"] = fic.title                  # get title
        info_dict["author"] = fic.author                # get author
        info_dict["hits"] = fic.hits                    # get hits
        info_dict["kudos"] = fic.kudos                  #get kudos
        info_dict["tags"] = fic.tags                    # get tags
        info_dict["score"] = round(top_n[count][0],4)
        info_dict["influential_words"] = top_n[count][2]
        count += 1
        top_n_fanfics.append(info_dict)
    
    #filter by tags
    if len(user_input_tags) != 0:
        top_n_fanfics = filter_fanfics(top_n_fanfics, user_input_tags)
    return top_n_fanfics
        
def getExtraFanficInfo(fanfic_id):
    info_dict = {}
    fic = Fanfics.query.filter_by(id=fanfic_id).first()
    info_dict['tags'] = fic.tags
    info_dict['fanfic_id'] = fic.id
    return [info_dict]

def user_description_search(user_description):
    """
    Uses SVD and cosine similarity between a description inputted by the user and 
    each webnovel to find the five most similar webnovels. 

    Argument(s):
    user_description:str - the description typed by the user

    Return(s):
    match: Dict{str:str, str:str} - a dictionary with the webnovel that most matches the user description
    """
    vectorizer = TfidfVectorizer()
    titles = []
    descriptions = []
    for n in Novels.query.all():
        titles.append(n.title)
        descriptions.append(n.description)
    docs_tfidf = vectorizer.fit_transform(descriptions)

    svd = TruncatedSVD(n_components=50)
    docs_svd = svd.fit_transform(docs_tfidf)
    user_tfidf = vectorizer.transform([user_description])
    user_svd = svd.transform(user_tfidf)
        
    sims = cosine_similarity(user_svd, docs_svd).flatten()
    result_indices = np.argsort(sims)
    matches = []
    for i in range(1,6):
        result_index = result_indices[-i]
        matches.append({'title': titles[result_index], 
                        'descr': descriptions[result_index]})
    return matches

def sql_search(query):
    """ Searches the webnovel database for a matching webnovel to the user typed query 
    using string matching.  
    Called for every character typed in the search bar.

    Argument(s):
    query:str - what the user types when searching for a webnovel

    Return(s):
    matches: [Dict{str: str}] - a list of matching webnovel dictionaries to the query. 
    Each dictionary includes the webovel title and description currently.  
    """
    novels = Novels.query.filter(Novels.title.ilike(f'%{query}%')).all()
    matches = [{"title": n.title, "descr": n.description} for n in novels]
    return matches


def getNovel(selectedNovel):
    novel = Novels.query.filter_by(title=selectedNovel).first()
    returnDict = {'title': novel.title,
                'descr':novel.description,
                'author':novel.author,
                'genres': novel.genres}
    return returnDict
        




    
