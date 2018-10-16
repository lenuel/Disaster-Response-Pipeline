
### Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overviewt<a name="overview"></a>
In this project,  disaster data from Figure Eight were analized to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. The trained machine learning pipeline categorizes these events so that messages can be send to an appropriate disaster relief agency.

The web app displays visualizations of the data, allows to input message and address it to appropriate category.

## Installation <a name="installation"></a>

I use Anaconda with Python version 3.6 to create this ap with the following libraries Pandas, NumPy, Matplotlib, Scikit-Learn, Flask etc. Additionally I use Plotly library to create interactive visuals on web page.

To install this package with conda run:
conda install -c plotly plotly


## Instructions to run<a name="results"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thanks Udacity and Figure Eight for this project.
