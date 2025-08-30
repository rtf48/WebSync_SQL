import os
from flask_sqlalchemy import SQLAlchemy




db = SQLAlchemy()


class Novels(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), index=True)
    description = db.Column(db.Text)
    author = db.Column(db.String(255))
    genres = db.Column(db.JSON)

class Cossims(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    webnovel_title = db.Column(db.String(255), index=True)
    similar_fic_titles = db.Column(db.JSON)

class Fanfics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), index=True)
    description = db.Column(db.Text)
    author = db.Column(db.String(255))
    hits = db.Column(db.Integer)
    kudos = db.Column(db.Integer)
    tags = db.Column(db.JSON)
    idx = db.Column(db.Integer,index=True)
    popularity = db.Column(db.Float)