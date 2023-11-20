# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit_feeling():
    feeling = request.form.get('feeling')

    print(f'User entered feeling: {feeling}')

    # MODEL

    # Load the model
    with open('StressDetectionMain/StressDetection.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Load the CountVectorizer
    with open('StressDetectionMain/CountVectorizer.pkl', 'rb') as file:
        loaded_cv = pickle.load(file)

    # Test with user input
    user=feeling
    
    data = loaded_cv.transform([user]).toarray()
    output = loaded_model.predict(data)
    print(output)

    result_message = "It seems you might be going through a tough time. Consider talking to someone you trust or seeking professional help." if output[0] == 1 else "Great to hear that you are feeling well!"

    return render_template('index.html',result_message=result_message)  # You can render the same template or redirect as needed

if __name__ == '__main__':
    app.run(debug=True)
