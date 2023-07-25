from flask import Flask, render_template, jsonify, request
from flask import Flask, render_template, request, session, redirect, url_for
from flask_wtf.csrf import CSRFProtect
import plotly.graph_objects as go
import plotly as plotly
from functionality.user import User
from functionality.database import database
from functionality.result_updater import result_updater
import functionality.tasks as tasks
import json
import importlib.util
from celery import Celery
import time
import redis
import os
import pandas as pd

# Things we've implemented today that need to be checked:
# - Model creation with duplicate params to an existing model
# - Checking the existing models params to see if there's a match
# - Handling the duplicate model


app = Flask(__name__, template_folder='static/templates', static_folder='static')

app.secret_key = 'to_the_moon'

celery = Celery('tasks', broker='redis://localhost:6379/0', task_serializer='pickle')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    my_db = database()
    my_db.get_all_usernames()
    users = my_db.users

    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        username = request.form['username']
        password = request.form['password']
        # TODO: Make this return the thing and keep the reg form out
        if username in users:
            return 'Username already exists!'
        my_db.add_user(first_name, last_name, username, password)
        users = my_db.users
        app.logger.debug(users)
        session['user_id'] = username
        app.logger.debug(f'{session["user_id"]} account created and logged in!')
        return redirect(url_for('profile'))

    return render_template('register.html')

@app.route('/profile')
def profile():
  return render_template('profile.html')

@app.route('/test_func')
def test_func():

  test = tasks.start_model_runner.delay()

  return render_template('profile.html')

@app.route('/get_graph_data', methods=['GET', 'POST'])
def get_graph_data():
    strategy = request.json['strategy']
    my_db = database()
    try:
        data = my_db.make_data(strategy)
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'Strategy is training on historical data... Check back in a few minutes...'})

    return jsonify(data)

@app.route('/get_user_strategies', methods=['GET', 'POST'])
def get_user_strategies():
    my_db = database()
    user_strategies = my_db.get_user_strategies(session['user_id'])

    app.logger.debug(f'{session["user_id"]} strategies: {user_strategies}')

    return jsonify(user_strategies)


@app.route('/make_strategy', methods=['GET','POST'])
def make_strategy():
    strat_name = request.form.get('name')
    min_ev = request.form.get('min_ev')
    min_odds = request.form.get('min_odds')
    max_odds = request.form.get('max_odds')
    min_min_com = request.form.get('min_min_com')
    max_min_com = request.form.get('max_min_com')
    my_db = database()
    this_user = User(session['user_id'])

    if strat_name in my_db.get_all_user_strategies():
            # Return a JSON response indicating that the username is taken
            return jsonify({'status': 'error', 'message': 'Strategy name is already taken'})
    
    if strat_name:
        this_user.add_strategy_to_user(session['user_id'], strat_name)
        tasks.make_strategy.delay(name=strat_name, 
                                  min_ev=float(min_ev), 
                                  min_odds=float(min_odds), 
                                  max_odds=float(max_odds), 
                                  min_min_com=float(min_min_com), 
                                  max_min_com=float(max_min_com),
                                  num_epochs=10
                                  )
        return jsonify({'status': 'success', 'message': ' '})

    
    return render_template('strategy_maker.html')


@app.route('/login', methods=['GET', 'POST'])  
def login():
  if request.method == 'POST':
    username = request.form['username']
    password = request.form['password']
    
    # Check credentials and redirect to user page if valid
    my_db = database()
    login_result = my_db.check_login_credentials(username, password)
    app.logger.debug(f'Login Result: {login_result}')

    if login_result:
        session['user_id'] = username
        app.logger.debug(f'Login Result: {session["user_id"]}')
    return redirect(url_for('profile'))


  return render_template('login.html')



if __name__ == '__main__':
    app.run(port=8080, debug=True)
