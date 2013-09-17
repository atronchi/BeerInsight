from flask import render_template,request
from app import app
import os

@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html",footer_color='white')

@app.route('/about')
def about():
    slides = ['static/slides/'+i for i in sorted([a[2] for a in os.walk('app/static/slides')][0])]
    idx = range(len(slides))
    return render_template("about.html",
        slides = slides, idx = idx,
        footer_color='white'
        )

@app.route('/contact')
def contact():
    return render_template("contact.html",footer_color='white')


# Map of beer locations

import cPickle,gzip,json
with gzip.open('scraper/scrape_ratebeer.pklz','rb') as f: loc_data=cPickle.load(f)
@app.route('/rb_sanjose')
def rd_sanjose():
    return json.dumps(loc_data['locations'])


# Forms for login and user beer preference selection
# see http://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-iii-web-forms
# see http://twitter.github.io/typeahead.js/examples/
from flask import flash, redirect
from forms import LoginForm,UserInputForm
@app.route('/login', methods = ['GET', 'POST'])
def login():
    return render_template('login.html', footer_color='white',
        form = LoginForm())

@app.route('/user_input', methods = ['GET', 'POST'])
def user_input():
    return render_template('user_input.html', footer_color='white',
        form = UserInputForm())

# load BA beer data to import to forms autofill
import gzip,cPickle
with gzip.open('recommender/reviews.pklz2') as f: BA=cPickle.load(f)
BA_brewers = list(set([b[0] for b in BA['beers']]))[0:1000]
BA_beers = list(set([b[1] for b in BA['beers']]))
@app.route('/brewers')
def brewers():
    query = request.args.get('q','')
    print query
    if query=='':
        return json.dumps(BA_brewers[0:1000])
    else:
        return json.dumps(BA_brewers)
@app.route('/beers')
def beers():
    return json.dumps(BA_beers)


with gzip.open('app/gapi_key.pklz','rb') as f: gapi=cPickle.load(f)
@app.route('/maps')
def maps():
    return render_template("maps.html",footer_color='black',
        key = gapi['key'],
        form = UserInputForm())


# views for testing/debugging
@app.route('/test') 
def test():
    return render_template("divtest.html")

@app.route('/debug')
def debug():
    return render_template("debug.html",
        slides = ['static/slides/'+i for i in sorted([a[2] for a in os.walk('app/static/slides')][0])],
        pwd=os.getcwd()
        )

