from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer # from flask_cors import CORS

import string

# Initialize Flask app (using __name__)
app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess the dataset (fix backslash in path)
dataframe = pd.read_csv(r"C:\Frontend\Chattrapati_Shivaji_Maharaj_Biase_Dataset_Final4.csv", encoding="latin-1")

data = dataframe.where(pd.notnull(dataframe), '')

# Rename 'Label' column to 'Category' for consistency
if 'Label' in data.columns:
    data.rename(columns={'Label': 'Category'}, inplace=True)

# Ensure consistent capitalization for categories
data['Category'] = data['Category'].str.strip().str.title()

# Assign categories as biased (0) or non-biased (1)
data.loc[data['Category'] == 'Biased', 'Category'] = 0
data.loc[data['Category'] == 'Non-Biased', 'Category'] = 1
data['Category'] = data['Category'].astype(int)

# Define X and Y variables
X = data['Text']
Y = data['Category']

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Text preprocessing function with stemming
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

# Apply preprocessing to the text data
X = X.apply(preprocess_text)

# Transform text data into feature vectors using TF-IDF
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Support Vector Machine model
model = SVC()
model.fit(X_train_tfidf, Y_train)

# Predict and evaluate the model
Y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
print("Model accuracy:", accuracy)
print("Model F1 score:", f1)

# Frontend route to display index.html
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Backend route to handle prediction
@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    input_text = data.get('text', '')

    # Preprocess the input text
    input_text_preprocessed = preprocess_text(input_text)
    input_data_feature = vectorizer.transform([input_text_preprocessed])

    # Make a prediction for the input text
    prediction = model.predict(input_data_feature)

    # Prepare the response
    if prediction[0] == 1:
        response_text = "The text is Non-Biased"
    else:
        response_text = "The text is Biased"

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)
