from flask import Flask, Response

app = Flask(__name__)

@app.route('/xml')
def xml_response():
    xml_data = '<example>Hello, World!</example>'
    return Response(xml_data, mimetype='text/xml')

@app.route('/json')
def example():
    data = {'name': 'John', 'age': 30}
    return data  # Flask convertit le dictionnaire en JSON

@app.route('/')
def hello_world():
    return '<html><body>Hello World!</body></html>'

if __name__ == '__main__':
    app.run()
