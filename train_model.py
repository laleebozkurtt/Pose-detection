import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Veri setini yükle
df = pd.read_csv('egzersiz_veri.csv')

# Veriyi ön işleme (örneğin, kategorik verileri sayısal verilere dönüştürme)
encoder = LabelEncoder()
df['Fitness_Level'] = encoder.fit_transform(df['Fitness_Level'])
df['Goal'] = encoder.fit_transform(df['Goal'])
df['Target_Area'] = encoder.fit_transform(df['Target_Area'])

# Özellikleri ve etiketleri ayır
X = df[['Fitness_Level', 'Goal', 'Target_Area']]  # Özellikler
y = df['Exercise']  # Etiketler (Egzersizler)

# Eğitim ve test verisine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Modeli kaydet
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
