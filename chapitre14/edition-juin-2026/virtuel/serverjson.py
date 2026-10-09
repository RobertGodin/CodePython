from flask import Flask, json

app = Flask(__name__)

@app.route('/json')
def example():
    data = {'name': 'John', 'age': 30}
    return json.dumps(data)

if __name__ == '__main__':
    app.run()
