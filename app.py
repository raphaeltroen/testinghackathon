from flask import Flask, render_template, request, jsonify
from solve import process_link

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    link = data['link']
    result = process_link(link)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)