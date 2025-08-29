import os
import json
import numpy as np
import mysql.connector
import time

data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

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

with open(cossim_json_file_path, 'r') as file: 
    file_contents = json.load(file)
    cossims_and_influential_words = file_contents['cossims_and_influential_words']
    fic_popularities = file_contents['fanfic_id_to_popularity']
    webnovel_title_to_index = file_contents['webnovel_title_to_index']
    index_to_fanfic_id = file_contents['index_to_fanfic_id']
    tags_list = file_contents['tags_list']
        
        
fanfics = {}
fanfic_files = ['fanfic_G_2019_processed-pg1.json', 'fanfic_G_2019_processed-pg2.json', 'fanfic_G_2019_processed-pg3.json']
for file in fanfic_files:
    file = os.path.join(data_directory, file)
    with open(file, 'r') as f: 
        temp_fanfic_list = json.load(f)

        for fanfic_info in temp_fanfic_list:
            fanfics[fanfic_info['id']] = fanfic_info


"""========================== SQL Setup: ============================="""


# MySQL connection settings
HOST = "websync_db"
USER = "admin"        
PASSWORD = "admin"  
DB_NAME = "websync"

def setup():

    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            # Connect to MySQL server
            conn = mysql.connector.connect(
                host=HOST,
                user=USER,
                password=PASSWORD
            )
            break
        except mysql.connector.Error:
            print(f"MySQL not ready, retrying... ({attempt+1}/{max_attempts})")
            time.sleep(2)
    
    cursor = conn.cursor()

    # Create database if it doesn’t exist
    cursor.execute("SHOW DATABASES LIKE %s", (DB_NAME,))
    if cursor.fetchone():
        #print(f"Database '{DB_NAME}' already exists. Exiting importer.")
        #cursor.close()
        #conn.close()
        #return
        cursor.execute(f"DROP DATABASE {DB_NAME}")
    
    cursor.execute(f"CREATE DATABASE {DB_NAME}")
    cursor.execute(f"USE {DB_NAME}")

    # Create tables
    cursor.execute(f"""
        CREATE TABLE novels(
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255),
        description MEDIUMTEXT,
        author VARCHAR(255),
        genres JSON
    )
    """)
    for i in range (len(novel_titles)):
        author = novel_data[i]['authors']
        if type(author) == list and len(author) > 0:
            author = author[0]
        elif type(author) == str:
            author = author
        else:
            author = ""
        cursor.execute(f"INSERT INTO novels (title, description, author, genres) VALUES (%s, %s, %s, %s)",
         (novel_titles[i][0], 
         novel_descriptions[i], 
         author, 
         json.dumps(novel_data[i]['genres'])))

    conn.commit()

    # Cleanup
    cursor.close()
    conn.close()

