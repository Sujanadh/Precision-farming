from flask import Flask, render_template, request
import tensorflow
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)

classes = ['100','25','50','75','d']

# @app.route('/', methods = ['GET'])
# def main():
#     return render_template('index.html')


model = models.load_model('model.h5')
model.make_predict_function()
def predict_function(f_path, model):
    i = image.load_img(f_path, target_size = (224,224))
    i = image.img_to_array(i)
    i = np.expand_dims(i, axis = 0)
    pred = model.predict(i)
    result = classes[np.argmax(pred[0])]
    return result
    
    
    

@app.route('/submit', methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        f = request.files['my_image']
        f_path = './static' + f.filename
        f.save(f_path)
        
        result = predict_function(f_path, model )
        return render_template('index.html', prediction = result)
    
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host=f"0.0.0.0:{PORT}",debug=False)
    #app.run(host="0.0.0.0",debug = False)
    
