/*
    beerinsight.js
*/

var map;
function initialize() {
    // initialize the map canvas
    var mapOptions = {
      center: new google.maps.LatLng(37.3894, -122.0819), // convention is N and E
      zoom: 10,
      mapTypeId: google.maps.MapTypeId.ROADMAP
    };
    map = new google.maps.Map(document.getElementById("map-canvas"),
        mapOptions);

    logoutUser();
    prefetchBrewers();
    $.getJSON('/rb_sanjose', function(data){ placeMarkers(data); });
}
google.maps.event.addDomListener(window, 'load', initialize);

var markers = [];

// place small markers with descriptive popups at all the locations
function placeMarkers(data,markerSize,markerColor) {
      // clear previous markers
      if (markers.length>0) { $.each(markers, function(){this.setMap(null);} ); markers = []; }

      // set defaults
      markerSize = markerSize || 1;
      markerColor= markerColor || 'black';

      $.each(data, function(){ 
          var contentString = '<div><a href="'+this.url+'"><h1>'+this.name+'</h1></a><br>' +
                              'address: '+this.addr+'<br>' + 
                              'telephone: '+this.tel+'<br>' + 
                              'hours: '+this.hours+'<br>';
          if (this.prediction != undefined) {
              var beercount = 0;
              if (this.beers.length>2) { beercount=3; } else { beercount=this.beers.length; }
              contentString += 'top recommended beers: <ol type=1>';
              for (var i=0; i<beercount; i++) {
                  contentString += '<li><a href="'+this.beers[i].item.url+'">'+this.beers[i].item.brewery+', '+this.beers[i].item.name+'</a><br>' +
                                   this.beers[i].item.style+', predicted rating='+this.beers[i].item.prediction.toFixed(1)+', ' +
                                   '<i>last seen '+this.beers[i].date+'</i></li>';
              }
              contentString += '</ol>';
          } else {
              contentString += 'recent beers: <ul>';
              $.each(this.beers, function(){ 
                  contentString += '<li><a href="'+this.item.url+'">'+this.item.brewery+', '+this.item.name+'</a><br>' +
                                   this.item.style+', RB score='+this.item.rb_score+', ' +
                                   '<i>last seen '+this.date+'</i></li>';
              });
              contentString += '</ul>';
          }
          contentString += '</div>';
          var infowindow = new google.maps.InfoWindow({content: contentString});
          var marker = new google.maps.Marker({
              position: new google.maps.LatLng(this.latlng[0], this.latlng[1]),
              title: this.name,
              icon: { // for more options: https://developers.google.com/maps/documentation/javascript/overlays#Symbols
                path:google.maps.SymbolPath.CIRCLE,
                scale:markerSize,fillColor:markerColor,fillOpacity:1,strokeColor:'black',strokeWeight:1
              }, map: map
              }); 
          markers.push(marker);
          google.maps.event.addListener(marker, 'click', function() {
              infowindow.open(map,marker);
              });
      }); // end each
};
      
/*
    prefetches for typeahead.js
*/

function prefetchBrewers() {
    var brewer_url = "/brewers";
    console.log('initializing typeahead for: '+brewer_url);

    $('#brewer').typeahead({                                
        name: 'brewer',
        local:[], //prefetch:{'url':brewer_url, 'ttl':2000},
        remote:brewer_url+'?q=%QUERY'
    });
};

function prefetchBeer() {
    var beer_url = "/beers?brewer=" + $('#brewer').val();

    /* need to both destroy and use a new name so the old backing store isn't used
    or use no name (implemented)
    https://github.com/twitter/typeahead.js/issues/41 */
    $('#beer').typeahead('destroy');

    $('#beer').typeahead({ prefetch:beer_url });
};


/*
    Functions to handle user ratings
*/

var myRatings = [];

function addRating() {
    myRatings.push( [
        $('#brewer').val(),
        $('#beer').val(),
        $('#rating').val()
        ] );

    if (user != '') { saveData(user); }

    printRatings();
};

function delRating(i) {
    myRatings.splice(i,1);

    if (user != '') { saveData(user); }

    printRatings();
};

function printRatings() {
    if (myRatings.length>0) {
        var ratings_html = "<table class='table' style='font-size:10pt;'>" +
             "<tr style='font-weight:bold;'><td>Brewery</td><td>Beer name</td><td>my rating</td><td></td></tr>";
        i=myRatings.length; while (i--) {
            ratings_html += '<tr> <td>'+myRatings[i][0]+'</td><td>'+myRatings[i][1]+'</td><td>'+myRatings[i][2]+'</td>' +
                            '<td><a href="#" onclick="delRating('+i+')">X</a></td> </tr>';
        }
        ratings_html += '</table>';
        $("#ratings_table").html(ratings_html);
    } else {
        $("#ratings_table").html("");
    }

    showRecBtn();
};


/*
    Functions to handle user login/logout and data
*/

var user = '';
function loginUser() {
    user = $('#user').val();
    if (user != '') {
        var content = "Logged in. " +
            "<a href='#' onClick='logoutUser();' style='padding-right:10px;'>Not "+user+"?</a> ";
            //"<a href='#' onClick='saveData();' class='btn btn-primary btn-sm'>Save data</a>"
        $("#login-status").html(content);

        // load user data if available
        $.getJSON('/user?name='+user+'&loaddata=True',
        function(data) {
            if (data.success) { 
                console.log('loaded data for user '+user);

                if (data.dat != undefined) {
                    myRatings=JSON.parse(data.dat);

                    // also plot markers at top 5 locations and show recommendations if done
                    if (data.loc != undefined) {
                        placeMarkers( data.loc.slice(0,5), 10,'green' );
                    }
                    if (data.beer != undefined) {
                        showRecommendations(data.beer.slice(0,10));
                    }
                }
            } else { 
                console.log('error loading data for user '+user+': '+data.errmsg);
                if (data.errmsg=='new user' && myRatings.length>0) { 
                    saveData(); 
                } else {
                    console.log('no ratings to save');
                }
            }

            printRatings();

        }); // end getJSON
    }
};

function logoutUser() {
    var content = "<input id='user' name='user' size='10' class='typeahead' style='font-size:10pt;' type='text'> " +
                  "<input class='btn btn-primary btn-xs' type='submit' value='Sign In'>";
    $("#login-status").html(content);

    // clear user name and ratings on logout
    user='';
    $("#user").val(user);

    myRatings=[];
    printRatings();

    console.log('user logged out');
};

function saveData() {
    myRatings_json = JSON.stringify(myRatings);
    $.getJSON('/user?name='+user+'&savedata='+myRatings_json, 
    function(data) {
        if (this.success) { 
            console.log('saved user data: '+data.success);
        } else { 
            console.log('error saving user data: '+data.success);
        }
    });
};


/*
    Functions to handle the recommender
*/

function runRecommender() {

    var content = "<p> Looking for some beers you will like.<br>" +
                  "This can take a couple minutes. <img src='static/img/spinning-wheel.gif'>" +
                  "<img src='static/img/beerinsight-logo-lg.png' style='max-height:25%;'></p>";
    $("#rec-content").html(content);
    showRecPane();

    console.log('getting recommendations for user '+user);

    $.getJSON('/user?name='+user+'&recommend=True', function(data) {
            console.log('got recommendations');

            showRecommendations(data.beer.slice(0,10));
            placeMarkers( data.loc.slice(0,5), 10,'green' );
        }
    );
};


// show beer recommendations
function showRecommendations(beer) {
    var content = "<div style='font-size:14px;'><p>The top locations with these beers are shown with <span style='color:green;'>green</span> markers on the map.</p>" +
                  '<p>Your top recommended beers and predicted ratings are:<ol type=1>';
    $.each(beer, function() { 
        content += "<li><a href='"+this.url+"'>"+this.brewery+', '+this.name+'</a> (<i>rating of '+this.prediction.toFixed(1)+'</i>)</li>'; 
        });
    content += '</ol></p></div>';
    $("#rec-content").html(content);

    document.getElementById('myrec-button').style.visibility='visible';
    showRecPane();
}


// conditionally show recommend button
function showRecBtn() {
    if (myRatings.length==0 || user=='') { 
        $("#rec-button").html('Enter at least one rating and your username to get recommendations.');
    } else {
        $("#rec-button").html("<a href='#' class='btn btn-primary btn-sm' onclick='runRecommender();'>Recommend</a>");
    }
}

// display the panes
function showRecPane() {
    document.getElementById('form-canvas').style.visibility='hidden';
    document.getElementById('recommend-canvas').style.visibility='visible';
}
function showRatPane() {
    document.getElementById('form-canvas').style.visibility='visible';
    document.getElementById('recommend-canvas').style.visibility='hidden';
}

