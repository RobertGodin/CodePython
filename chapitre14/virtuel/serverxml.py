from flask import Flask, Response

app = Flask(__name__)

@app.route('/xml')
def xml_response():
    xml_data = '<example>Hello, World!</example>'
    return Response(xml_data, mimetype='text/xml')

if __name__ == '__main__':
    app.run(debug=True)

