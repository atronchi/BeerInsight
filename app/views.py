from flask import render_template
from app import app
import os

@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html",
        title = 'BeerSuggest',
        )

@app.route('/about')
def about():
    slides = ['static/slides/'+i for i in sorted([a[2] for a in os.walk('app/static/slides')][0])]
    idx = range(len(slides))
    return render_template("about.html",
        title = 'BeerSuggest',
        slides = slides, idx = idx
        )

@app.route('/contact')
def contact():
    return render_template("contact.html",
        title = 'BeerSuggest'
        )


# import pickled ratebeer data
rb_file = 'scrape_ratebeer.pklz'
import pickle,gzip
with gzip.open(rb_file,'rb') as f:
    loc_data=pickle.load(rb_file)

@app.route('/maps')
def maps():
    return render_template("google-maps.html",
        title = 'BeerSuggest'
       ,locations = ...
        )

from flask import flash, redirect
from forms import LoginForm
@app.route('/login', methods = ['GET', 'POST'])
def login():
    form = LoginForm()
    return render_template('login.html', 
        title = 'Sign In',
        form = form)


# views for testing/debugging
@app.route('/test') 
def test():
    return render_template("divtest.html",
        title = 'BeerSuggest'
        )

@app.route('/debug')
def debug():
    return render_template("debug.html",
        title = 'BeerSuggest',
        slides = ['static/slides/'+i for i in sorted([a[2] for a in os.walk('app/static/slides')][0])],
        pwd=os.getcwd()
        )

