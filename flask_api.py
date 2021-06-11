from flask import Flask
import os
from flask import request, jsonify

model_path = os.path.join("results", "001-pgan-carpet-4l-1024-preset-v2-1gpu-fp32", "network-final.pkl")

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/random", methods=['POST'])
def random():
    type = request.get_json()["type"]
    if type == "torkaman":
        return jsonify({"image": type})


app.run(debug=True)