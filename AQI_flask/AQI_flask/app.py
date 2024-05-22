from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
import os                                               
import time                        
import numpy as np                 
import pandas as pd                
import matplotlib   
import seaborn as sns 
matplotlib.use('agg')
from tensorflow.keras.models import load_model
hypermodel=load_model('single_step_lstm.keras')

def get_sin_cos_timestamp(timestamp):
    day = 24*60*60
    year = (365.2425)*day

    date_time = pd.to_datetime(timestamp, format='%d.%m.%Y %H:%M:%S')
    timestamp = pd.Timestamp.timestamp(date_time)
    
    return [np.sin(timestamp * (2 * np.pi / day)),
            np.cos(timestamp * (2 * np.pi / day)),
            np.sin(timestamp * (2 * np.pi / year)),
            np.cos(timestamp * (2 * np.pi / year))]

def auto_predict(model, date_range):
    train_mean=np.load('mean.npy')
    train_std=np.load('std.npy')
    y_future = []
    X_future=np.load('X_future.npy')
    index = 0

    for target_date in date_range:
        # Make new prediction
        prediction = model.predict(X_future[-1:], verbose=None)
        y_future.append(prediction)

        # Create new input
        input = X_future[-1][1:]
        num_features = 1
        timestamp_sin_cos = get_sin_cos_timestamp(target_date)
        timestamp_sin_cos = (timestamp_sin_cos - train_mean) / train_std
        
        observation = np.concatenate((prediction[0], timestamp_sin_cos), axis=0)
        observation = np.expand_dims(observation, axis=0)

        input = np.concatenate((input, observation), axis=0)
        input = np.expand_dims(input, axis=0)
        
        X_future = np.concatenate((X_future, input), axis=0)

        print(f"{index+1}/{len(date_range)}", end='\r', flush=True)

        index += 1
        
    return np.array(y_future)

def denormalize(data):
    return (data * 23.063405) + 25.067993



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    if request.method == 'POST':
        date_start = request.form['date_start']
        date_end = request.form['date_end']

        # Generate a plot based on the input dates
        future_dates = pd.date_range(start='2023-04-01', end=date_end, freq='5H')
        predictions = auto_predict(hypermodel, future_dates)
        predictions=denormalize(predictions)
        for i,j in enumerate(predictions[:,0,0]):
            if j<0:
                predictions[:,0,0][i]=0
        future_df = pd.DataFrame(predictions[:,0,0], index = future_dates, columns =['Future Prediction'])
        new=future_df.loc[str(date_start)+' '+'00:00:00':]
        # Generate a plot based on the input dates
        plt.figure()
        sns.lineplot(data=new, x=new.index, y=new['Future Prediction'], label="Future Prediction(PM2.5)")
        plt.xlabel('Date')
        plt.ylabel('PM2.5')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('index.html', plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
