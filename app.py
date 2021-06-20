from flask import Flask
from infer import model_answer

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/qa/<string:question>")
def qa(question):
    return {
        "answer": model_answer(question),
    }
