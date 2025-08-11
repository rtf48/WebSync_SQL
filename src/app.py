import json
import os
from flask import Flask, render_template, request, session
from flask_cors import CORS
from backend.helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import copy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from backend.scratch import edit_distance_search, filter_fanfics, get_svd_tags
from backend.scratch import insertion_cost, deletion_cost, substitution_cost
# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

data_directory = os.path.join(current_directory, 'backend', 'data')

# Specify the path to the JSON file relative to the current script
novel_json_file_path = os.path.join(data_directory, 'novel_info.json')

cossim_json_file_path = os.path.join(data_directory, 'webnovel_to_fanfic_cossim.json')

"""========================== Gathering data: ============================="""
def getKeyInfo(data,key):
    lst = []
    for i in range(len(data)):
        lst.append(data[i][key])
    return lst

def getTitleInfo(data):
    lst = []
    for i in range(len(data)):
        lst.append(data[i]['titles'][0])
    return lst

with open(novel_json_file_path, 'r') as file:
    novel_data = np.array(json.load(file))
    novel_titles = getKeyInfo(novel_data,'titles')
    novel_descriptions = getKeyInfo(novel_data,'description')
    novel_title_to_index = {}
    for i in range (len(novel_titles)):
        novel_title_to_index[novel_titles[i][0]] = i
        
fanfics = {}
fanfic_files = ['fanfic_G_2019_processed-pg1.json', 'fanfic_G_2019_processed-pg2.json', 'fanfic_G_2019_processed-pg3.json']
for file in fanfic_files:
    file = os.path.join(data_directory, file)
    with open(file, 'r') as f: 
        temp_fanfic_list = json.load(f)

        for fanfic_info in temp_fanfic_list:
            fanfics[fanfic_info['id']] = fanfic_info

with open(cossim_json_file_path, 'r') as file: 
    file_contents = json.load(file)
    cossims_and_influential_words = file_contents['cossims_and_influential_words']
    fic_popularities = file_contents['fanfic_id_to_popularity']
    webnovel_title_to_index = file_contents['webnovel_title_to_index']
    index_to_fanfic_id = file_contents['index_to_fanfic_id']
    tags_list = file_contents['tags_list']

app = Flask(__name__)

app.secret_key = 'BAD_SECRET_KEY_1'
CORS(app)

""" ========================= Backend stuff: ============================"""
""" Global variable to store all the user's input tags"""

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
    print("a1. In json_search(query) in app.py          No app.route()")
    matches = []
    titles = set()
    for i in range (len(novel_titles)):
        for j in range(len(novel_titles[i])):
            novel_title_copy = novel_titles[i][j].lower().replace(u"\u2019", "'")
            if query.lower() in novel_title_copy and query != "" and novel_titles[i][0] not in titles:
                matches.append({'title': novel_titles[i][0],'descr':novel_descriptions[i]})
                titles.add(novel_titles[i][0])
    return matches

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

@app.route("/fanfic-recs/")
def recommendations(): 
    """
    Called when the user clicks "Show Reccommendations"
    Links to showResults(title) in base.html
    """
    print("a2. In recomendations() app.py           app.route(/fanfic-recs/)")
    title = request.args.get('title')
    weight = request.args.get("popularity_slider")
    results = webnovel_to_top_fics(title, 49, int(weight)/100)
    count = 0
    while len(results) < 10 and count <= 400:
        results = webnovel_to_top_fics(title, 100 + count, int(weight)/100)
        count += 50
    return results

def webnovel_to_top_fics(webnovel_title, num_fics, popularity_weight):
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
    print("a3. In webnovel_to_top_fanfictions() app.py         No app.route()")
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
    user_input_tags = request.args.get("tags").split(",")
    if '' in user_input_tags:
        user_input_tags.remove('')
    
    if len(user_input_tags) != 0:
        top_n_fanfics = filter_fanfics(top_n_fanfics, user_input_tags)
    return top_n_fanfics
    
def getExtraFanficInfo(fanfic_id):
    info_dict = {}
    info_dict['tags'] = fanfics[fanfic_id]['tags']
    info_dict['fanfic_id'] = fanfic_id
    return [info_dict]

@app.route("/")
def home():
    print("a4. In home() in app.py          app.route(/)")
    return render_template('home.html', title="")

@app.route("/results")
def results():
    """ Called when the user clicks the --> arrow on the home page."""
    print("a5. In results() in app.py           app.route(/results)")
    return render_template('base.html', webnovel_title=request.args.get("title"))
    

@app.route("/titleSearch")
def titleSearch():
    """
    Gets the user typed query, and calls json_search to return relevant webnovels.
    Links to function filterText(id) in home.html.
    """
    print("a6. In titleSearch() in app.py.          app.route(/titleSearch)")
    text = request.args.get("inputText")
    return json_search(text)

@app.route("/descrSearch")
def descrSearch():
    """
    Gets the user typed query, and calls json_search to return relevant webnovels.
    Links to function filterText(id) in home.html.
    """
    print("a6. In descrSearch() in app.py.          app.route(/descrSearch)")
    text = request.args.get("inputText")
    return user_description_search(text)

@app.route("/getNovel")
def getNovel():
    """
    Retrieves the selected webnovel title, author, description, and genres from the second page.
    Links to function setup() in base.html
    Called as soon as user enters the second page

    Returns: 
    returnDict: Dict{
        title: webnovel title
        descr: webnovel description
        author: The first listed author of the webnovel
        genres: All the genres of the webnovel
    }
    """
    print("a8. In getNovel() in app.py          app.route(/getNovel)")

    selectedNovel = request.args.get("title")
    index = novel_title_to_index[selectedNovel]
    returnDict = {'title': novel_titles[index],
                  'descr':novel_descriptions[index],
                  'author':novel_data[index]['authors'][0],
                  'genres': novel_data[index]['genres']}
    return returnDict

@app.route("/inforeq")
def getExtraInfo():
    print("a11. in getExtraInfo() in app.py().            app.route(/inforeq) ")
    fanfic_id = int(request.args.get("fanfic_id"))
    return getExtraFanficInfo(fanfic_id)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)