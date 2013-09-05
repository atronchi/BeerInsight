#!/usr/bin/env python

import os

from flask import Flask
from flask import url_for
from flask import render_template
from flask import jsonify

app = Flask(__name__)

# WTForms config per: http://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-iii-web-forms
app.config.from_object('config')

from app import views

