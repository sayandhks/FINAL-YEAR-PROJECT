from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import io
import base64
import os
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from gtts import gTTS
from tensorflow.keras.models import load_model
from playsound import playsound

# Load the model
hypermodel = load_model('single_step_lstm.keras')


def get_sin_cos_timestamp(timestamp):
    day = 24 * 60 * 60
    year = (365.2425) * day

    date_time = pd.to_datetime(timestamp, format='%d.%m.%Y %H:%M:%S')
    timestamp = pd.Timestamp.timestamp(date_time)

    return [np.sin(timestamp * (2 * np.pi / day)),
            np.cos(timestamp * (2 * np.pi / day)),
            np.sin(timestamp * (2 * np.pi / year)),
            np.cos(timestamp * (2 * np.pi / year))]


def auto_predict(model, date_range):
    train_mean = np.load('mean.npy')
    train_std = np.load('std.npy')
    y_future = []
    X_future = np.load('X_future.npy')
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

        print(f"{index + 1}/{len(date_range)}", end='\r', flush=True)

        index += 1

    return np.array(y_future)


def denormalize(data):
    return (data * 23.063405) + 25.067993


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    peak_value = None
    peak_date = None

    if request.method == 'POST':
        date_start = request.form['date_start']
        date_end = request.form['date_end']

        # Generate future dates based on the input dates
        future_dates = pd.date_range(start='2023-04-01', end=date_end, freq='5H')
        predictions = auto_predict(hypermodel, future_dates)
        predictions = denormalize(predictions)
        for i, j in enumerate(predictions[:, 0, 0]):
            if j < 0:
                predictions[:, 0, 0][i] = 0
        future_df = pd.DataFrame(predictions[:, 0, 0], index=future_dates, columns=['Future Prediction'])
        new = future_df.loc[str(date_start) + ' ' + '00:00:00':]

        # Find peak PM2.5 value and corresponding date
        peak_value = new['Future Prediction'].max()
        peak_date = new[new['Future Prediction'] == peak_value].index[0].strftime('%Y-%m-%d %H:%M:%S')

        # Generate a plot based on the input dates
        plt.figure()
        sns.lineplot(data=new, x=new.index, y=new['Future Prediction'], label="Future Prediction(PM2.5)")
        plt.xlabel('Date')
        plt.ylabel('PM2.5')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Convert peak date and value to speech so it's advisable to stay indoors, use air purifiers, wear masks when outdoors, limit outdoor activities, and stay informed about air quality advisories to minimize exposure to harmful pollutants.
        tts_text = f"The peak PM2.5 value of {peak_value} was observed on {peak_date}. so it's advisable to stay indoors, use air purifiers, wear masks when outdoors, limit outdoor activities, and stay informed about air quality advisories to minimize exposure to harmful pollutants."
        tts = gTTS(text=tts_text, lang='en')
        # Define the folder path where you want to save the file
        output_folder = "C:/Users/Gatha Reghunath/Desktop/voice/templates"
        # Save the audio file
        tts.save(os.path.join(output_folder, "peak_pm25.mp3"))

        # Play the saved audio file
        playsound(os.path.join("C:/Users/Gatha Reghunath/Desktop/voice/templates/peak_pm25.mp3"))

        # Run whatsapp.py
        subprocess.Popen(["python", "whatsapp.py"])

    return render_template('index.html', plot_url=plot_url, peak_value=peak_value, peak_date=peak_date)


if __name__ == "__main__":
    app.run(debug=True)
