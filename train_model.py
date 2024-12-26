import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Veri Setini Yükle
data = pd.read_csv("egzersiz_veri.csv")

# Kategorik Verileri Dönüştür
label_encoder = LabelEncoder()
data['Fitness Level'] = label_encoder.fit_transform(data['Fitness Level'])
data['Goal'] = label_encoder.fit_transform(data['Goal'])
data['Target Area'] = label_encoder.fit_transform(data['Target Area'])

X = data[['Fitness Level', 'Goal', 'Target Area']]
y = data['Exercise']  # Y hedefi sadece 'Exercise'

# Modeli Eğit
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Modeli Kaydet
pickle.dump(model, open("model.pkl", "wb"))
