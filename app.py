#!flask/bin/python
from flask import Flask, request, render_template    #記得要import render_template
import pandas as pd
from transformers import BertTokenizer
import numpy as np
import tensorflow as tf #tf2.0.0 版本 不然怪怪的
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
def bert_encode(data,maximum_length) :
    input_ids = []
    attention_masks = []
    encoded = tokenizer.encode_plus(      
            data,#data.text[i]
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,     
            return_attention_mask=True,)
        
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)
def create_model(bert_model):
  input_ids = tf.keras.Input(shape=(60,),dtype='int32')
  attention_masks = tf.keras.Input(shape=(60,),dtype='int32')
  
  output = bert_model([input_ids,attention_masks])
  output = output[1]
  output = tf.keras.layers.Dense(32,activation='relu')(output)
  output = tf.keras.layers.Dropout(0.2)(output)

  output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
  model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
  model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
  return model
bert_model = TFBertModel.from_pretrained('bert-large-uncased')
model = create_model(bert_model)
model.summary()
model.load_weights('my_model_weights.h5')
print('載入模型成功',model)

app = Flask(__name__)

#網頁執行/時，會導至index.html
@app.route('/')#, methods=['GET']
def getdata():
    return render_template('index.html')


#index.html按下submit時，會取得前端傳來的username，並回傳"此推文:"+test+predict
@app.route('/', methods=['POST'])
def submit():
    test = request.form.get('username')
    test_input_ids,test_attention_masks = bert_encode(test,60)
    result = model.predict([test_input_ids,test_attention_masks])
    result = np.round(result).astype(int)
    if(result[0][0]==1):
        predict='為災難推文'
    else:
        predict='非災難推文'
    
    return "此推文:"+test+predict
    
 

if __name__ == '__main__':
    app.run(debug=True)

    