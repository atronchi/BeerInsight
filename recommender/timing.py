import time

# timing functions
def tic():
    return time.time()

def checkTime(num,i): # print timing message
    global toc1
    global toc2
    if tic()-toc1 > 5:
        toc1 = tic() # reset timer

        num_togo = num-(i+1)
        elapsed = ( tic()-toc2 )/3600. # hrs
        remain = elapsed/(i+1)*num_togo

        rem_str = '{0}'.format(num_togo).rjust(7) + ' togo, ' + \
                  '{0:.2f}'.format(elapsed) + ' hrs elapsed, ' + \
                  '{0:.2f}'.format(remain ) + ' hrs togo'

        print '\b'*(len(rem_str)+2),
        print rem_str,; sys.stdout.flush()

