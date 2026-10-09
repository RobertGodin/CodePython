from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/json')
def example():
    data = {'name': 'John', 'age': 30}
    return jsonify(data)

if __name__ == '__main__':
    app.run()
