from flask import (
    Flask, request, redirect
)
import models
import time

app = Flask(__name__)

@app.route("/", methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        data = request.get_json()
        question = data['msg']
        answer = dict()
        start = time.time()
        # infer should be here
        end = time.time()
        answer['answerType'] = "dialog"
        answer['elapsedTime'] = end - start
        answer['msg'] = "임시 답변"
        return answer
    
    return redirect('http://127.0.0.1/')

