function initialize() {
    geocoder = new google.maps.Geocoder();
    var mapOptions = {
      center: new google.maps.LatLng(37.3894, -122.0819), // convention is N and E
      zoom: 12,
      mapTypeId: google.maps.MapTypeId.ROADMAP
    };

    var map = new google.maps.Map(document.getElementById("map-canvas"),
        mapOptions);

    jQuery.getJSON('/rb_sanjose', function(data) {
      jQuery.each(data, function(){ 
          //map.setCenter(results[0].geometry.location); // don't need to reset the map center
          //console.log('loc: '+this.latlng[0]+','+this.latlng[1]);
          var contentString = '<div><a href="'+this.url+'"><h1>'+this.name+'</h1></a><br>' +
                              'address: '+this.addr+'<br>' + 
                              'telephone: '+this.tel+'<br>' + 
                              'hours: '+this.hours+'<br>' +
                              'beers:<ul>';
          jQuery.each(this.beers, function(){ contentString += 
                              '<li><a href="'+this.item.url+'">'+this.item.brewery+', '+this.item.name+'</a><br>' +
                              this.item.style+', RB score='+this.item.rb_score+', ' +
                              '<i>last seen '+this.date+'</i></li>'; });
          contentString += '</ul></div>';
          var infowindow = new google.maps.InfoWindow({
              content: contentString
              });
          var marker = new google.maps.Marker({
              //position: this.latlng,
              position: new google.maps.LatLng(this.latlng[0], this.latlng[1]),
              title: this.name,
              icon: { 
                // for more options: https://developers.google.com/maps/documentation/javascript/overlays#Symbols
                path:google.maps.SymbolPath.CIRCLE,
                scale:3,fillColor:'red',fillOpacity:1,strokeColor:'black',strokeWeight:1
              },
              map: map
              }); 
          google.maps.event.addListener(marker, 'click', function() {
              infowindow.open(map,marker);
              });
          }); // end each
      }); // end getJSON
          
    prefetchBrewer();
    logoutUser();
} // end initialize

google.maps.event.addDomListener(window, 'load', initialize);
  function prefetchBrewer() {
    var brewer_url = "/brewers";
    console.log('initializing typeahead for: '+brewer_url);

    $('#brewer').typeahead({                                
      name: 'brewer',
      local:[], //prefetch:{'url':brewer_url, 'ttl':2000},
      remote:brewer_url+'?q=%QUERY'
    });
  };

  var beer_cache_no=0;
  function prefetchBeer() {
    var beer_url = "/beers?brewer=" + $('#brewer').val();

    /* need to both destroy and use a new name so the old backing store isn't used
    https://github.com/twitter/typeahead.js/issues/41 */
    $('#beer').typeahead('destroy');

    $('#beer').typeahead({
      beer:'name'+beer_cache_no,
      prefetch:beer_url
    });
    beer_cache_no += 1;
  };



var myRatings = [];
function addRating() {
    myRatings.push( [
        $('#brewer').val(),
        $('#beer').val(),
        $('#rating').val()
        ] );
    printRatings();
};
function delRating(i) {
    myRatings.splice(i,1);
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
  };


  function runRecommender() {
  };


  var user = '';
  function loginUser() {
    user = $('#user').val();
    if (user != '') {
        var content = "Logged in. " +
            "<a href='#' onClick='logoutUser();' style='padding-right:10px;'>Not "+user+"?</a> " +
            "<a href='#' onClick='saveData();' class='btn btn-primary btn-sm'>Save data</a>";
        $("#login-status").html(content);

        loadData();
    }
  };
  function logoutUser() {
    var content = "<input id='user' name='user' size='10' class='typeahead' style='font-size:10pt;' type='text'> " +
                  "<input class='btn btn-primary btn-sm' type='submit' value='Sign In'>";
    $("#login-status").html(content);

    user='';
    $("#user").val(user);
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
  function loadData() {
    $.getJSON('/user?name='+user, 
    function(data) {
        if (data.success) { 
            console.log('loaded user data: '+data.success);

            if (data.dat != undefined) {
                myRatings=JSON.parse(data.dat);
                printRatings();
            }
        } else { 
            console.log('error loading user data: '+data.success);
        }
    });
  };

