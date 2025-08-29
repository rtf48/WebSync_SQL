import os
import json
import numpy as np
import copy
from flask_sqlalchemy import SQLAlchemy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from backend.scratch import edit_distance_search, filter_fanfics
from backend.scratch import insertion_cost, deletion_cost, substitution_cost

from backend.sql_setup import novel_title_to_index, novel_titles, novel_descriptions, novel_data
from backend.sql_setup import webnovel_title_to_index, cossims_and_influential_words, fic_popularities, index_to_fanfic_id, fanfics
import backend.sql_setup as sql_setup
from backend.models import Novels


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
    webnovel_index = webnovel_title_to_index[webnovel_title]
    sorted_fanfics_tuplst = cossims_and_influential_words[str(webnovel_index)]
    # top_n = np.copy(sorted_fanfics_tuplst[:num_fics])
    top_n = copy.deepcopy(sorted_fanfics_tuplst[:num_fics])
    max_pop = np.max(list(fic_popularities.values())) / 10
    for fic_tuple in top_n:
        fic_tuple[0] = fic_popularities[str(int(fic_tuple[1]))] / max_pop * popularity_weight + fic_tuple[0] * (1 - popularity_weight)
        top_n = sorted(top_n, key=lambda x: x[0], reverse=True)[:num_fics]
        top_n_fanfic_indexes = [t[1] for t in top_n]
        top_n_fanfics = []

        count = 0
        for i in top_n_fanfic_indexes:
            fanfic_id = index_to_fanfic_id[str(int(i))]
            info_dict = {}
            info_dict["fanfic_id"] = fanfic_id                              # get fanfic id
            info_dict["description"] = fanfics[fanfic_id]['description']    # get description
            info_dict["title"] = fanfics[fanfic_id]["title"]                # get title
            info_dict["author"] = fanfics[fanfic_id]["authorName"]          # get author
            info_dict["hits"] = fanfics[fanfic_id]["hits"]                  # get hits
            info_dict["kudos"] = fanfics[fanfic_id]["kudos"]                # get kudos
            info_dict["tags"] = fanfics[fanfic_id]["tags"]                  # get tags
            info_dict["score"] = round(top_n[count][0],4)
            info_dict["influential_words"] = top_n[count][2]
            count += 1
            top_n_fanfics.append(info_dict)
        
        
        if len(user_input_tags) != 0:
            top_n_fanfics = filter_fanfics(top_n_fanfics, user_input_tags)
        return top_n_fanfics
        
def getExtraFanficInfo(fanfic_id):
    info_dict = {}
    info_dict['tags'] = fanfics[fanfic_id]['tags']
    info_dict['fanfic_id'] = fanfic_id
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
    docs_tfidf = vectorizer.fit_transform(novel_descriptions)

    svd = TruncatedSVD(n_components=50)
    docs_svd = svd.fit_transform(docs_tfidf)
    user_tfidf = vectorizer.transform([user_description])
    user_svd = svd.transform(user_tfidf)
        
    sims = cosine_similarity(user_svd, docs_svd).flatten()
    result_indices = np.argsort(sims)
    matches = []
    for i in range(1,6):
        result_index = result_indices[-i]
        matches.append({'title': novel_titles[result_index][0], 
                        'descr': novel_descriptions[result_index]})
    return matches

def json_search(query):
    """ Searches the webnovel database for a matching webnovel to the user typed query 
    using string matching.  
    Called for every character typed in the search bar.

    Argument(s):
    query:str - what the user types when searching for a webnovel

    Return(s):
    matches: [Dict{str: str}] - a list of matching webnovel dictionaries to the query. 
    Each dictionary includes the webovel title and description currently.  
    """
    matches = []
    titles = set()
    for i in range (len(novel_titles)):
        for j in range(len(novel_titles[i])):
            novel_title_copy = novel_titles[i][j].lower().replace(u"\u2019", "'")
            if query.lower() in novel_title_copy and query != "" and novel_titles[i][0] not in titles:
                matches.append({'title': novel_titles[i][0],'descr':novel_descriptions[i]})
                titles.add(novel_titles[i][0])
    return matches


def getNovel(selectedNovel):
    novel = Novels.query.filter_by(title=selectedNovel).first()
    returnDict = {'title': novel.title,
                'descr':novel.description,
                'author':novel.author,
                'genres': novel.genres}
    return returnDict
        




    
