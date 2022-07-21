from flask import Flask, request
from flask_cors import CORS, cross_origin
import os

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

@app.route('/')
@cross_origin()
def home():
    return "hello"

@app.route('/save', methods=['POST'])
@cross_origin()
def save():
    # save
    data = request.get_json()
    print('data =',data)
    dat = data.get('dat')
    i = data.get('i')

    # write file
    root = './draw_iou/rotated_rect'
    try:
        int(i)
    except:
        pass
    filename = os.path.join(root, str(i) + '.txt')
    with open(filename, 'w') as f:
        f.write(str(dat))
    print('saved', i)
    return 'saved'

if __name__ == '__main__':
    app.run(debug=True, port=8000)
