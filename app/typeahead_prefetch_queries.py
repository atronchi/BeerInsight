import sqlite3
db_name = 'app/typeahead_prefetch.sqlite'

def fetch_brewers(q=''):
    # open database connection
    with sqlite3.connect(db_name) as con:
        cur = con.cursor()

        if q=='':
            print 'no brewer query specified, returning null set...'
            return []
        else:
            return [ i[0] for i in cur.execute('''
                select distinct
                    brewer
                from tpf
                where brewer like ?;
                ''',('%'+q+'%',)).fetchall() ]

def fetch_beers(brewer,q=''):
    # open database connection
    with sqlite3.connect(db_name) as con:
        cur = con.cursor()

        if brewer=='':
            print 'no brewer query specified, returning null set...'
            return []

        if q=='':
            return [ i[0] for i in cur.execute('''
                select
                    beer
                from tpf
                where brewer=?
                ''',(brewer,)).fetchall() ]
        else:
            return [ i[0] for i in cur.execute('''
                select
                    beer
                from tpf
                where brewer=? and beer like ?
                ''',(brewer,'%'+q+'%')).fetchall() ]

