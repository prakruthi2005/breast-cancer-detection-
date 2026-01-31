
from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained CNN model
cnn_model = load_model('model/cnn_model.h5')

# Load and train classical models
def preprocess_data():
    df = pd.read_csv('cancer.csv')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)
    X = np.array(df.drop(['classes'], axis=1))
    y = np.array(df['classes'])
    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return sc, pca, knn, svm

X, y = preprocess_data()
scaler, pca_model, knn_model, svm_model = train_models(X, y)

# Predict using CNN
def predict_xray_image(filepath):
    img = image.load_img(filepath, target_size=(150, 150), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = cnn_model.predict(img_array)[0][0]
    return "Malignant" if prediction > 0.5 else "Benign"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.endswith('.csv'):
                data = pd.read_csv(filepath)
                data_scaled = scaler.transform(data)
                data_pca = pca_model.transform(data_scaled)
                prediction = knn_model.predict(data_pca)
                result = f"KNN CSV Prediction: {prediction.tolist()}"

            elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                prediction = predict_xray_image(filepath)
                result = f"X-ray Image Prediction (CNN): {prediction}"

            else:
                result = "Unsupported file format."

        else:
            try:
                features = [float(request.form.get(f'f{i}')) for i in range(1, 10)]
                input_data = np.array(features).reshape(1, -1)
                input_scaled = scaler.transform(input_data)
                input_pca = pca_model.transform(input_scaled)
                prediction = svm_model.predict(input_pca)
                result = f"SVM Prediction: {prediction[0]}"
            except:
                result = "Invalid manual input."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
