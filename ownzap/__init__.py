from flask import  Flask
from datetime import datetime
import tensorflow as tf

model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights='ownzap/static/hello40.hdf5',
    input_tensor=None,
    input_shape=(150,150,1),
    pooling='avg',
    classes=10
    )
model.compile(
    loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy']
    )

app=Flask(__name__)
UPLOAD_FOLDER = 'ownzap/static/profile_pics'
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif','.webp','.tiff','.psd']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from ownzap import routes

