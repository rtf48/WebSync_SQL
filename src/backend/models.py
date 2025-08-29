import os
from flask_sqlalchemy import SQLAlchemy




db = SQLAlchemy()


class Novels(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255))
    description = db.Column(db.Text)
    author = db.Column(db.String(255))
    genres = db.Column(db.JSON)