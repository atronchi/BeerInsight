#!/bin/bash

while [ /bin/true ]; 
do
    sudo twistd -n web --port 80 --wsgi run_flask.app -l twistd.log
done


