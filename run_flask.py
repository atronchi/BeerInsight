#!/usr/bin/python

from app import app

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    #port = int(os.environ.get('PORT', 5000))
    #app.run(host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=80,debug=True) # must run as root to use ports below 1024
    #app.run(debug=True)
   

