from flask import Flask, render_template, request
import pickle
import pandas as pd

# Flask uygulamasını başlat
app = Flask(__name__)

# Modeli yükle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Egzersiz önerilerini yap
@app.route('/recommend', methods=['POST'])
def recommend():
    # Kullanıcıdan gelen veriyi al
    fitness_level = request.form['fitness_level']
    goal = request.form['goal']
    target_area = request.form['target_area']

    # Veriyi model için hazırlamak
    user_input = pd.DataFrame([[fitness_level, goal, target_area]],
                            columns=['Fitness_Level', 'Goal', 'Target_Area'])
    
    # Modeli kullanarak egzersizleri tahmin et
    recommended_exercises = model.predict(user_input)

    # Sonuçları frontend'e ilet
    return render_template('result.html', exercises=recommended_exercises)

if __name__ == '__main__':
    app.run(debug=True)
