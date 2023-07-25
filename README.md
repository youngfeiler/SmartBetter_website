# SmartBetter_website
website 


Set up:
- Open a terminal
- Navigate to directory where app.py is located
- activate the python env where torch works 
- pip install celery
- pip install redis
- in a different terminal start your redis server, command should be: redis-server but could cause errors
- in that first terminal, start the celery worker: celery -A functionality.tasks worker --loglevel=info --concurrency=4
- run app.py in vscode or whatever using the same environtment where torch works and all that
- download mlb_data folder from google drive an put it in this repo


- navigate to http://127.0.0.1:8080/

- easier said than done
