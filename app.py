from flask import Flask, render_template, request, jsonify
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize stemmer
ps = PorterStemmer()

app = Flask(__name__)

# Load the model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorization.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl or vectorization.pkl not found!")
    print("Please make sure both files are in the same directory as app.py")

def transform_text(text):
    """
    Preprocess text the same way as training data:
    1. Convert to lowercase
    2. Tokenize
    3. Keep only alphanumeric characters
    4. Remove stopwords and punctuation
    5. Apply stemming
    """
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the message from the form
        message = request.form.get('message', '')
        
        if not message.strip():
            return jsonify({
                'error': 'Please enter a message'
            })
        
        # Preprocess the text (IMPORTANT: same as training!)
        transformed_message = transform_text(message)
        
        # Transform the message using the vectorizer
        message_vector = vectorizer.transform([transformed_message])
        
        # Convert sparse matrix to dense array (important for models trained on dense data)
        message_vector_dense = message_vector.toarray()
        
        # Make prediction
        prediction = model.predict(message_vector_dense)[0]
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(message_vector_dense)[0]
            confidence = max(prediction_proba) * 100
        except:
            confidence = None
        
        # Determine the result
        if prediction == 1:
            result = "SPAM"
            result_class = "spam"
        else:
            result = "NOT SPAM"
            result_class = "not-spam"
        
        return jsonify({
            'result': result,
            'result_class': result_class,
            'confidence': confidence,
            'message': message
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
