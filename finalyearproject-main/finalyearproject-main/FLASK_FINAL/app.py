import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)


classes = ['100% Field Capacity', '25% Field Capacity',
           '50% Field Capacity', '75% Field Capacity', 'DISEASED']

model = models.load_model('tomatoes.h5')
# model.summary()


model.make_predict_function()


def predict_function(f_path, model):
    i = image.load_img(f_path, target_size=(224, 224))
    i = image.img_to_array(i)
    i = np.expand_dims(i, axis=0)
    pred = model.predict(i)
    result = np.argmax(pred[0])
    confidence = round(100 * (np.max(pred[0])), 2)
    return result, confidence


# routes
@app.route("/", methods=['GET'])
def main():
    return render_template("index.html")


@app.route("/login", methods=['GET', 'POST'])
def login():
    return render_template('form.html')


@app.route("/about", methods=['GET', 'POST'])
def about():
    return render_template('about.html')


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    img = request.files['my_image']
    img_path = './static/test_img/' + img.filename
    img.save(img_path)
    result , confidence = predict_function(img_path, model)
    if result == 0: 
        result = '100% Field Capacity'
        recom = '२५% सिचाइ कम गर्नु होला । अहिले मल राख्नु पर्दैन । १५ दिन पछी फेरी जाच गर्नु होला । '
        return render_template("index.html", prediction=result ,confidence = confidence ,recommendation=recom, img_path=img_path)
    
    elif result == 1: 
        result = '25% Field Capacity'
        recom = 'पानी को मात्र निकै कमी छ । ५०% पानी अझै बढाउनु होला । समय मा पानी हाल्ने  गर्नु। '
        return render_template("index.html", prediction=result ,confidence = confidence ,recommendation=recom, img_path=img_path)
 
    elif result == 2: 
        result = '50% Field Capacity'
        recom = '२०% देखी २५% सिचाइ बढाउनु होला र १५ दिन पछी फेरी जाच गर्नु होला । '
        return render_template("index.html", prediction=result ,confidence = confidence ,recommendation=recom, img_path=img_path)
    
    elif result == 3: 
        result = '75% Field Capacity'
        recom = 'दुबै सिचाइ र मल को मात्र एक्दम उचित रहेको छ । '
        return render_template("index.html", prediction=result ,confidence = confidence ,recommendation=recom, img_path=img_path)
    
    elif result == 4: 
        result = 'DISEASED LEAF'
        recom = 'बिरुवा मा रोग लगेको छ । कृपया समयमै रोग को पत्ता लगाइ फेरी जाच गर्नु होला । '
        return render_template("index.html", prediction=result ,confidence = confidence ,recommendation=recom, img_path=img_path)

if __name__ == '__main__':
    app.run(host=f"0.0.0.0:{PORT}",debug=False)
