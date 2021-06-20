from flask import Flask, url_for
from infer import model_answer

app = Flask(__name__)

@app.route("/")
def routes_info():
    links = {}
    for rule in app.url_map.iter_rules():
        defaults = rule.defaults if rule.defaults is not None else ()
        arguments = rule.arguments if rule.arguments is not None else ()
        if "GET" in rule.methods and len(defaults) >= len(arguments):
            url = url_for(rule.endpoint, **(rule.defaults or {}))
            links[url] = rule.endpoint
    return links

@app.route("/qa/<string:question>")
def qa(question):
    return {
        "answer": model_answer(question),
    }
