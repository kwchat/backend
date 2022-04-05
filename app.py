from flask import (
    Flask, request, jsonify, redirect
)
import models

app = Flask(__name__)

@app.route("/", methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        data = request.get_json()
        question = data.msg
        answer = dict()
        # infer should be here
        answer['answerType'] = "normal"
        answer['elapsedTime'] = 0
        answer['msg'] = "임시 답변"
        return jsonify(answer)
    
    return redirect('http://127.0.0.1/')

