from flask import render_template,request
import app.recommender
from app import app # this should come after the other app imports
import os
import cPickle,gzip,json

# load data, one or more of these takes a damn long time
with gzip.open('app/scrape_ratebeer.pklz','rb') as f: loc_data=cPickle.load(f)
with gzip.open('app/typeahead_prefetch.pklz2','rb') as f: tpf=cPickle.load(f)
with gzip.open('app/gapi_key.pklz','rb') as f: gapi=cPickle.load(f)


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



# Forms for login and user beer preference selection
# see http://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-iii-web-forms
# see http://twitter.github.io/typeahead.js/examples/
from flask import flash, redirect
from forms import LoginForm,UserInputForm
import os,sys

@app.route('/rb_sanjose')
def rb_sanjose():
    return json.dumps(loc_data['locations'])

@app.route('/brewers')
def brewers():
    q = request.args.get('q','')
    print 'getting brewers like '+q
    if q=='':
        return json.dumps(tpf['brewer_prefetch'])
    else:
        BA_brewers = list(set( [b for b in tpf['brewers_n_beers'].keys() if b.startswith(q)] ))
        return json.dumps(BA_brewers)

@app.route('/beers')
def beers():
    brewer = request.args.get('brewer','')
    q = request.args.get('q','')

    print 'getting beers for brewer: '+brewer
    if brewer in tpf['brewers_n_beers'].keys():
        BA_beers = list(set( [b for b in tpf['brewers_n_beers'][brewer] if b.startswith(q)] ))
    else:
        BA_beers = []
    return json.dumps(BA_beers)

@app.route('/user')
def user():
    uname = request.args.get('name','')
    savedata = request.args.get('savedata','')
    loaddata = request.args.get('loaddata','')
    recommend= request.args.get('recommend','')

    if uname != '':
        print 'received data for user '+uname+': ',type(savedata),savedata
        fnam = 'app/userdata/'+uname+'.pklz2'

        if savedata != '':
            try:
                print 'writing '+fnam
                with gzip.open(fnam,'wb') as f: cPickle.dump({'webdat':savedata},f,protocol=2)
                return json.dumps({'success':True})

            except:
                print 'fail'
                return json.dumps({'success':False,'errmsg':'error writing user data'})

        if loaddata=='True':
            if os.path.exists(fnam):
                try:
                    print 'opening '+fnam
                    with gzip.open(fnam,'rb') as f: dat=cPickle.load(f)
                    if 'beers' in dat.keys() and 'locations' in dat.keys():
                        return json.dumps({'success':True,
                                           'dat':dat['webdat'],
                                           'beer':dat['beers'],
                                           'loc':dat['locations']
                                          })
                    else:
                        return json.dumps({'success':True,
                                           'dat':dat['webdat']
                                          })
                except:
                    print 'fail'
                    return json.dumps({'success':False,'errmsg':'error loading user data'})

            else: 
                print 'new user: no data to load'
                return json.dumps({'success':False,'errmsg':'new user'})

        if recommend=='True':
            import sys
            try:
                print 'running recommender...'
                import app.recommender as rec
                (beer,loc) = rec.run(uname)
                return json.dumps({'success':True,'beer':beer,'loc':loc})
            except:
                print 'recommender error: ',sys.exc_info()[0]
                return json.dumps({'success':False,'errmsg':'recommender error'})

    else:
        print 'no username supplied in query'
        return json.dumps({'success':False,'errmsg':'no username specified'})


@app.route('/maps', methods=['GET','POST'])
def maps():
    return render_template("maps.html",footer_color='black',
        key = gapi['key'],
        form = UserInputForm()
        )


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

