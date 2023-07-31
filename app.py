from flask import Flask, render_template, jsonify, request
from flask import Flask, render_template, request, session, redirect, url_for
from flask_wtf.csrf import CSRFProtect
import plotly.graph_objects as go
import plotly as plotly
from functionality.user import User
from functionality.database import database
from functionality.result_updater import result_updater
import functionality.tasks as tasks
from functionality.util import american_to_decimal
import pandas as pd
from functionality.tasks import celery


def create_app():
    app = Flask(__name__, template_folder='static/templates', static_folder='static')
    app.secret_key = 'to_the_moon'
    app.config['SESSION_COOKIE_DURATION'] = 0
    app.secret_key = 'to_the_moon'
    app.celery = celery
    app.config['SESSION_COOKIE_DURATION'] = 0
    return app

app = create_app()


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/profile')
def profile():
  if 'user_id' in session:
        return render_template('profile.html')
  else:
        return redirect(url_for('login'))


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
        phone = '+1' + str(request.form['phone_number'])
        if username in users:
            return render_template('register.html', username_exists=True, form_data=request.form)
        else:
            my_db.add_user(first_name, last_name, username, password, phone)
            users = my_db.users
            return redirect(url_for('make_strategy'))
    return render_template('register.html', username_exists=False, form_data={})


@app.route('/login', methods=['GET', 'POST'])  
def login():
  if request.method == 'POST':
    username = request.form.get('username')
    password = request.form.get('password')
    my_db = database()
    login_allowed = my_db.check_login_credentials(username, password)
    app.logger.debug(f'Login Result: {login_allowed}')
    if login_allowed:
        session['user_id'] = username
        return redirect(url_for('profile'))
    elif not login_allowed:
        app.logger.debug("Invalid credentials. Rendering login page...")
        return render_template('login.html', incorrect_password=True, form_data=request.form)


  return render_template('login.html')

@app.route('/information')  
def information():
  return render_template('information.html')

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

@app.route('/team_dist_data', methods=['GET', 'POST'])
def team_dist_data():

    strategy = request.args.get('strategy')

    app.logger.debug(f'HERE: {strategy}')
    my_db = database()
    try:
        data = my_db.make_team_dist_data(strategy)
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': '<a style:"color: white;">Strategy is training on historical data... Check back in a few minutes...</a>'})

    return data

@app.route('/book_dist_data', methods=['GET', 'POST'])
def book_dist_data():
    strategy = request.args.get('strategy')
    my_db = database()
    try:
        data = my_db.make_book_dist_data(strategy)
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'cant make the data'})

    return data

@app.route('/active_bets', methods=['GET', 'POST'])
def active_bets():
    strategy = request.args.get('strategy')
    app.logger.debug(strategy)
    my_db = database()
    data = my_db.make_active_bet_data(strategy)
    try:
        data = my_db.make_active_bet_data(strategy)
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'cant make the data'})

    return data

@app.route('/get_user_strategies', methods=['GET', 'POST'])
def get_user_strategies():
    my_db = database()
    user_strategies = my_db.get_user_strategies(session['user_id'])

    app.logger.debug(f'{session["user_id"]} strategies: {user_strategies}')

    return jsonify(user_strategies)

@app.route('/make_strategy', methods=['GET','POST'])
def make_strategy():
  if 'user_id' not in session:
        return redirect(url_for('login'))
        
  
  if request.method == 'POST':
    form_data = request.form
    form_data_dict = dict(form_data)
    bettable_books_item = request.form.getlist('bettable_books')[0]
    bettable_books = bettable_books_item.split(',')
    strat_name = form_data_dict['name']
    min_ev = 10
    min_odds = american_to_decimal(form_data_dict['min_odds'])
    max_odds = american_to_decimal(form_data_dict['max_odds'])
    min_min_com = form_data_dict['min_min_com']
    max_min_com = form_data_dict['max_min_com']
    my_db = database()
    this_user = User(session['user_id'])
    if strat_name in my_db.get_all_user_strategies():
            return jsonify({'status': 'error', 'message': 'Strategy name is already taken'})
    
    if strat_name:
        this_user.add_strategy_to_user(session['user_id'], strat_name)
        tasks.make_strategy.delay(name=strat_name, 
                                  min_ev=float(min_ev), 
                                  min_odds=float(min_odds), 
                                  max_odds=float(max_odds), 
                                  min_min_com=float(min_min_com), 
                                  max_min_com=float(max_min_com),
                                  bettable_books = bettable_books,
                                  num_epochs=int(100)
                                  )
        return jsonify({'status': 'success', 'message': ' '})

    
  return render_template('strategy_maker.html')

@app.route('/check_if_text_allowed', methods=['GET','POST'])
def check_if_text_allowed():
    strategy_name = request.form.get('strategy')

    username =session['user_id']

    database_instance = database()

    result = database_instance.check_text_permission(username, strategy_name)

    return jsonify({'allowed': result})

# in development 
@app.route('/update_text_alert', methods=['GET','POST'])
def update_text_alert():

    data = request.get_json() 

    strategy_name =  data.get('strategy')

    is_checked = data.get('isChecked')

    username = session['user_id']

    database_instance = database()

    result = database_instance.update_text_permission(username, strategy_name)

    return jsonify({'message': result})

 

if __name__ == '__main__':
    app.run()
