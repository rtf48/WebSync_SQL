import json
import os
from flask import Flask, render_template, request, session
from flask_cors import CORS
from backend.helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import backend.search as search
import numpy as np
import copy


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))



app = Flask(__name__)

app.secret_key = 'BAD_SECRET_KEY_1234'
CORS(app)


@app.route("/fanfic-recs/")
def recommendations(): 
    """
    Called when the user clicks "Show Reccommendations"
    Links to showResults(title) in base.html
    """
    title = request.args.get('title')
    weight = request.args.get("popularity_slider")

    user_input_tags = request.args.get("tags").split(",")
    if '' in user_input_tags:
        user_input_tags.remove('')
    
    results = search.webnovel_to_top_fics(title, 49, int(weight)/100, user_input_tags)
    count = 0
    while len(results) < 10 and count <= 400:
        results = search.webnovel_to_top_fics(title, 100 + count, int(weight)/100, user_input_tags)
        count += 50
    return results



@app.route("/")
def home():
    return render_template('home.html', title="")

@app.route("/results")
def results():
    """ Called when the user clicks the --> arrow on the home page."""
    return render_template('base.html', webnovel_title=request.args.get("title"))
    

@app.route("/titleSearch")
def titleSearch():
    """
    Gets the user typed query, and calls json_search to return relevant webnovels.
    Links to function filterText(id) in home.html.
    """
    text = request.args.get("inputText")
    return search.json_search(text)

@app.route("/descrSearch")
def descrSearch():
    """
    Gets the user typed query, and calls json_search to return relevant webnovels.
    Links to function filterText(id) in home.html.
    """
    text = request.args.get("inputText")
    return search.user_description_search(text)

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

    selectedNovel = request.args.get("title")
    return search.getNovel(selectedNovel)

@app.route("/inforeq")
def getExtraInfo():
    fanfic_id = int(request.args.get("fanfic_id"))
    return search.getExtraFanficInfo(fanfic_id)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)

