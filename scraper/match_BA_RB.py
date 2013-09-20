import gzip,cPickle
import numpy as np

def loadData(fnam):
    with gzip.open(fnam,'rb') as f: return cPickle.load(f)

BA = loadData('recommender/reviews.pklz2')
RB = loadData('scraper/scrape_ratebeer.pklz')

BA_beers = BA['beers']
RB_beers = np.array([ [i['brewery'],i['name']] for i in RB['beers'] ])

# generate index mapping each RB entry to the appropriate BA entry
RB_to_BA = np.zeros( len(RB_beers), dtype=int )


import ngram
BA_brewerbeers = ngram.NGram(enumerate(
    [b[0]+'|'+b[1] for b in BA_beers]
    ), key=lambda x:x[1].lower() )
RB_matches = []
for i in enumerate(RB_beers):
    RB_matches.append(
        BA_brewerbeers.find( 
            (i[1][0]+'|'+i[1][1]).lower() 
            ))
    print (i[0],i[1][0]+'|'+i[1][1])
    print RB_matches[-1]
    
import gzip,cPickle
with gzip.open('match_BA_RB.pklz2','wb') as f: 
    cPickle.dump(RB_matches,f,protocol=2)


# to write beer lists to files
def writeLists():
    import codecs
    with codecs.open('BA_beer_list.txt','w','utf-8') as f:
        for i in enumerate(BA_beers):
            f.write( unicode(str(i[0]) + ' | ' + i[1][0] + ' | ' + i[1][1] + '\n') )
    with codecs.open('RB_beer_list.txt','w','utf-8') as f:
        for i in enumerate(RB_beers):
            f.write( unicode(str(i[0]) + ' | ' + i[1][0] + ' | ' + i[1][1] + '\n') )


'''
# array of manually matched RB to BA
RB_to_BA = [
 [0,3533], 
 [1,21092],
 [2,62145],
 [3,12098],
 [4,56838],
 [5,-1],
 [6,52917],
 [7,1890],
 [8,1828],
 [9,29380],
 [10,58820],
 [11,43231],
 [12,-1],
 [13,57993],
 [14,42614],
 [15,5006],
 [16,32973],
 [17,27880],
 [18,20018],
 [19,31007],
 [20,3176],
 [21,32984],
 [22,20099],
 [23,15832],
 [24,-1],
 [25,12062],
 [26,-1],
 [27,-1],
 [28,43200],
 [29,-1],
 [30,9740],
 [31,36672],
 [32,11302],
 [33,-1],
 [34,51718],
 [35,42605],
 [36,-1],
 [37,55784],
 [38,-1],
 [39,-1],
 [40,-1],
 [41,-1],
 [42,1894],
 [43,19186],
 [44,39668],
 [45,22700],
 [46,52034],
 [47,-1],
 [48,46938],
 [49,-1],
 [50,-1],
 [51,1834],
 [52,8032],
 [53,20088],
 [54,42773],
 [55,63371],
 [56,39706],
 [57,22462],
 [58,17307],
 [59,61780],
 [60,32988],
 [61,49464],
 [62,-1],
 [63,-1],
 [64,-1],
 [65,51151],
 [66,12518],
 [67,1133],
 [68,6687],
 [69,56002],
 [70,27872],
 [71,1094],
 [72,19140],
 [73,-1],
 [74,22463],
 [75,-1],
 [76,-1],
 [77,-1],
 [78,32968],
 [79,20370],
 [80,51202],
 [81,22456],
 [82,-1],
 [83,21175],
 [84,61770],
 [85,51665],
 [86,34005],
 [87,53294],
 [88,-1],
 [89,3128],
 [90,2509],
 [91,2421],
 [92,27432],
 [93,20414],
 [94,11285],
 [95,8090],
 [96,39746],
 [97,26366],
 [98,22206],
 [99,-1],
 [100,56008],
 [101,48762],
 [102,3539],
 [103,37830],
 [104,61963],
 [105,-1],
 [106,39699],
 [107,22196],
 [108,50597],
 [109,36667],
 [110,3519],
 [111,-1],
 [112,-1],
 [113,-1],
 [114,5592],
 [115,1897],
 [116,47780],
 [117,55963],
 [118,56309],
 [119,58791],
 [120,39781],
 [121,337],
 [122,53252],
 [123,12921],
 [124,39498],
 [125,47634],
 [126,36698],
 [127,1055],
 [128,-1],
 [129,-1],
 [130,53310],
 [131,25074],
 [132,-1],
 [133,-1],
 [134,-1],
 [135,58806],
 [136,-1],
 [137,39776],
 [138,33941],
 [139,39780],
 [140,-1],
 [141,-1],
 [142,62139],
 [143,-1],
 [144,28407],
 [145,13151],
 [146,57973],
 [147,12909],
 [148,42086],
 [149,-1],
 [150,53325],
 [151,1134],
 [152,20333],
 [153,18111],
 [154,3561],
 [155,26589],
 [156,-1],
 [157,8170],
 [158,-1],
 [159,-1],
 [160,10719],
 [161,1832],
 [162,33939],
 [163,62167],
 [164,12953],
 [165,8056],
 [166,4347],
 [167,49700],
 [168,4351],
 [169,12885],
 [170,1054],
 [171,13073],
 [172,-1],
 [173,6143],
 [174,-1],
 [175,-1],
 [176,55907],
 [178,-1],
 [179,22868],
 [180,20332],
 [181,46827],
 [182,-1],
 [183,1003],
 [184,-1],
 [185,-1],
 [186,19431],
 [187,4556],
 [188,12318],
 [189,33314],
 [190,-1],
 [191,50621],
 [192,3580],
 [193,8190],
 [194,29578],
 [195,12048],
 [196,62165],
 [197,33941],
 [198,54999],
 [199,55805],
 [200,-1],
 [201,53332],
 [202,25057],
 [203,4575],
 [204,56299],
 [205,54806],
 [206,22198],
 [207,50591],
 [208,132],
 [209,20019],
 [210,6568],
 [211,46076],
 [212,-1],
 [213,23750],
 [214,-1],
 [215,42758],
 [216,-1],
 [217,8038],
 [218,-1],
 [219,39546],
 [220,12796],
 [221,-1],
 [222,58803],
 [223,50369],
 [224,-1],
 [225,52909],
 [226,18129],
 [227,2043],
 [228,-1],
 [229,-1],
 [230,14174],
 [231,47049],
 [232,33804],
 [233,-1],
 [234,58778],
 [235,22190],
 [236,-1],
 [237,-1],
 [238,57993],
 [239,26346],
 [240,-1],
 [241,34898],
 [242,19893],
 [243,12885],
 [244,-1],
 [245,-1],
 [246,46978],
 [247,-1],
 [248,55963],
 [249,43223],
 [250,22798],
 [251,-1],
 [252,-1],
 [253,1873],
 [254,19201],
 [255,36547],
 [256,53264],
 [257,12736],
 [258,39735],
 [259,52954],

'''