from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Keep only required columns
df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]

# Remove missing values
df.dropna(inplace=True)

# Convert Rating to numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Remove invalid rows
df.dropna(inplace=True)

# ==============================
# Create Hit/Flop Label
# ==============================

df['Result'] = np.where(df['Rating'] >= 7.0, 'Hit', 'Flop')

# ==============================
# Encode Text Columns
# ==============================

genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
actor1_encoder = LabelEncoder()
actor2_encoder = LabelEncoder()
actor3_encoder = LabelEncoder()

df['Genre'] = genre_encoder.fit_transform(df['Genre'])

df['Director'] = director_encoder.fit_transform(df['Director'])

df['Actor 1'] = actor1_encoder.fit_transform(df['Actor 1'])

df['Actor 2'] = actor2_encoder.fit_transform(df['Actor 2'])

df['Actor 3'] = actor3_encoder.fit_transform(df['Actor 3'])

# ==============================
# Lists for HTML Dropdowns
# ==============================

genres = sorted(genre_encoder.classes_)

directors = sorted(director_encoder.classes_)

actors1 = sorted(actor1_encoder.classes_)

actors2 = sorted(actor2_encoder.classes_)

actors3 = sorted(actor3_encoder.classes_)
# ==============================
# Train the Machine Learning Model
# ==============================

X = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ==============================
# Home Page
# ==============================

@app.route('/')
def home():

    return render_template(
        "movie.html",
        genres=genres,
        directors=directors,
        actors1=actors1,
        actors2=actors2,
        actors3=actors3,
        prediction=""
    )


# ==============================
# Prediction
# ==============================

@app.route('/predict', methods=['POST'])
def predict():

    genre = request.form['genre']
    director = request.form['director']
    actor1 = request.form['actor1']
    actor2 = request.form['actor2']
    actor3 = request.form['actor3']

    # Encode user input
    genre = genre_encoder.transform([genre])[0]
    director = director_encoder.transform([director])[0]
    actor1 = actor1_encoder.transform([actor1])[0]
    actor2 = actor2_encoder.transform([actor2])[0]
    actor3 = actor3_encoder.transform([actor3])[0]

    # Create dataframe for prediction
    movie = pd.DataFrame({
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor1],
        'Actor 2': [actor2],
        'Actor 3': [actor3]
    })

    # Predict
    prediction = model.predict(movie)[0]

    return render_template(
        "movie.html",
        genres=genres,
        directors=directors,
        actors1=actors1,
        actors2=actors2,
        actors3=actors3,
        prediction=prediction
    )


# ==============================
# Run Flask
# ==============================


if __name__ == "__main__":
    app.run(host="0.0.0.0")
