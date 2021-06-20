# Deep-chat Backend

## Getting Started
````
git clone https://github.com/kwchat/backend.git
cd backend
. venv/bin/activate
pip install -r requirements.txt
export FLASK_APP=app
flask run
# Then, request answer to localhost:5000/qa/<string:question>
# Then, we can get {"answer" : "Some answer model gives"}
````

