from flask import (
    Flask, request, redirect
)
from flask_cors import CORS
from models import ChatModel
import time
from config import *

app = Flask(__name__)
CORS(app)
kwchat = ChatModel()
answerTypes = ['dialog', 'drqa']

@app.route("/", methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        data = request.get_json()
        question = data['msg']
        start = time.time()
        answer = kwchat.predict([question])
        end = time.time()
        answer = [item for (item,) in answer] # unpack
        answer = {
            'answerType': answerTypes[int(answer[1])],
            'elapsedTime': end - start,
            'msg': answer[0],
        }
        return answer
    
    return redirect(webServerUrl)

