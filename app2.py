from flask import Flask, request, jsonify
import os
from caption_model import CaptionModel
import base64

# create the flask object
app = Flask(__name__)
caption_model = None


@app.route('/')
def index():
    return "Index Page"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['image']
    if data == None:
        return 'Got None'
    else:
        root_path = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(root_path, './images/', data.filename)
        data.save(image_path)
        response = caption_model.captioning(image_path)
        return jsonify(text=response)


@app.route("/test", methods=['POST'])
def test():
    data = request.form['image']
    root_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(root_path, './images/image.png')
    with open(image_path, "wb") as file:
        file.write(base64.decodebytes(data.encode('UTF-8')))
    response = caption_model.captioning(image_path)
    return jsonify(text=response)
    

if __name__ == "__main__":
    caption_model = CaptionModel()
    app.run(host='0.0.0.0', debug=False)
    # path=os.path.dirname(os.path.abspath(__file__))
    # path_image=os.path.join(path,'images',f'Test(2).jpg')
    # response= caption_model.captioning(path_image)
    # print(response)
