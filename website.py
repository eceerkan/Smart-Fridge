import os
import ast
import base64
import requests
from flask import Flask, render_template, request
import json

app = Flask(__name__)

# Helper function to get data from text file

def get_data():
    contents = f.read("FridgeContents.json")
    print(contents)
    data = json.loads(contents)
    print(data)
    return data

# Helper function to update data in text file
def update_data(item, day, month, new_item=None):
    data = get_data()
    if new_item:
        data[new_item] = data.pop(item)
        item = new_item
    data[item][0][2] = day
    data[item][0][3] = month
    new_contents = json.dumps(data)
    f.update("FridgeContents.json", new_contents)


# Route to render index.html template
@app.route('/')
def index():
    data = get_data()
    return render_template('website.html', data=data)

# Route to update data and render index.html template
# Route to update data and render index.html template
@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    item = data['item']
    day = data['day']
    month = data['month']
    old_item = data['old_item']
    old_day = data['old_day']
    old_month = data['old_month']
    if item != old_item:
        update_data(old_item, old_day, old_month, new_item=item)
    else:
        update_data(item, day, month)
    data = get_data()
    return json.dumps(data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
