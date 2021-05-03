# The Imports needed
import pandas as pd
from flask import Flask, request, render_template
from engine import RecommendationEngine
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)

#this dataframe contains all the movies and show available to rate.
df_content = pd.read_csv('static/df_content.csv', index_col=3)

# This will redirect user to the homepage which is called '/index', and show the user df_content dataframe
@app.route('/', methods=['POST','GET'])
def home():
    return render_template('index.html' ,tables=[df_content.to_html(classes='data')], titles=df_content.columns.values)

# This will add the user's input to the model which then the model will restart with the updated data
# and from then it will generate top 10 recommmended movies.
@app.route("/index", methods=['POST','GET'])
def ratings():
    content_id = request.form.getlist('content_name[]')
    rating = request.form.getlist('rating[]')
    logger.debug("User 500000 TOP ratings requested")
    ratings = recommendation_engine.add_ratings(content_id,rating)
    return render_template('simple.html',  tables=[ratings.to_html(classes='data')], titles=ratings.columns.values)

# Creating an instance of the Recommendation Engine
global recommendation_engine
recommendation_engine = RecommendationEngine('static/dataset2.csv', 'static/df_content.csv')

app.run(debug=True)
