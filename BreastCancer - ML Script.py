# UWAGA PROSZE PRZECZYTAC CALE
# Niniejsze dane są open-source i pochodzą z UCI - https://doi.org/10.24432/C51P4M.
# Są to informacje na temat nowotworów piersi pacjentek onkologicznych 
# Uniwersyteckiego Instytutu Onkologicznego w Ljubljanie w Słoweni.
# Z racji iż są to dane biologiczne, to w niniejszym projekcie zastosuję algorytmy uczenia maszynowego
# które następnie poddam walidacji za pomocą dwóch metod
# Train-Test Split (Mniej Dokładny) i Kroswalidacji (Bardziej Dokładny).

# Pamiętajmy co jest celem - Sprawdzenie czy pacjentka doswiadczy nawrotu, i jeśli, to która cecha
# będzie do tego prowadziła.

# W niniejszej pracy sa trzy modele: Las Losowy, Las Losowy (Po Pruningu) i Sieć Neuronowa.
#Przy pruningu i Sieci Neuronowej skrypt troszke dluzej sobie dziala zanim poda wyniki.

#Pobór wstępnych bibliotek

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.optimizers import AdamW  # Używamy AdamW zamiast Adam


#-----LAS LOSOWY-----

# 1) Wczytanie danych. Stworzenie bardziej klarownych nazw. Uporządkowanie danych z .CSV
file_path = "C:/Users/Mateusz/OneDrive/Desktop/Breast_Cancer_Data/breast-cancer.data"
column_names = ["Czy_Nawrot", "Wiek", "CzyMenopauza", "Guz_Wielkosc", "LiczbaZajetychWezlow", 
                "Czy_OtoczkaWezlaPrzerwana", "StopienZlosliwosci", "KtoraPiers", "Umiejscowienie", "CzyRadioterapia"]
df = pd.read_csv(file_path, names=column_names, delimiter=",")

# 2) Czyszczenie danych z błędów. Zamiana '?' na NaN. Następnie wypełnienie braków modą - czyli najczęściej występującą wartością
# dla danej kategorii.

for col in df.columns:
    df[col].replace("?", np.nan, inplace=True)
    df[col].fillna(df[col].mode()[0], inplace=True)  #Tutaj wypełniamy modą. Jest to jeden z "trików", kiedy są drobne luki w danych.

df = df.apply(lambda x: pd.factorize(x)[0]) #Tutaj zamiana wartości kategorycznych na liczby

# 3) Tutaj robimy podział danych na zbiór treningowy i testowy. Ogólna zasada jest taka, że zbiór treningowy powinien
# stanowić około 80% naszych danych, zaś zbiór testowy pozostałe 20%. Tak tez robimy.
# Staramy sie przewidziec ktora zmienna powoduje, ze na bank bedzie nawrot nowotworu.
# Czy to wiek, czy to miejsce, czy zlosliwosc, czy inna cecha - No wlasnie, co najbardziej wplywa na powrot choroby?

X = df.drop("Czy_Nawrot", axis=1)
y = df["Czy_Nawrot"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Trenowanie modelu - Uzywamy tutaj Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5) Ocena modelu. Tutaj tworzymy tabele niczym z programu Statistica. Pokazuje ona nam cechy naszego modelu.
print("Dokładność modelu:", accuracy_score(y_test, y_pred))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

#Komentarz do powyzej:
#Model ma wysoka czulosc (recall) na wykrywanie braku nawrotu (92%). Niska zas na wykrywanie nawrotu (29%).
#Precyzje sa na mniej-wiecej podobnym poziomie, oznacza to ze  kiedy juz typowana jest dana kategoria,
#to z prawdopodobienstwem na poziomie ok. 70% jest ona prawdziwa.
#Z racji iz to Nawrót jest dla nas najwazniejszy, model Lasow Losowych nie spelnia swojego zadania. Wykrywa tylko 29%
# wszystkich nawrotów. Kolejnym krokiem byłoby albo zastosowanie pruningu, albo innego modelu. Zanim jednak do tego dojdziemy -
# Wygenerujemy Macierz Pomylek.


# 6) Macierz pomyłek - Tak jak w testach na Covid i wszystkich innych biologicznych sprawach, jest ona absolutna koniecznościa.
# Obliczenie macierzy
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)

# Zastosowanie mapy kolorów "RdYlGn": zielony = dobre przewidywania, czerwony = błędy
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdYlGn', cbar=True,
                 xticklabels=["Brak nawrotu", "Nawrót"],
                 yticklabels=["Brak nawrotu", "Nawrót"])

plt.xlabel("Przewidziana klasa", fontsize=16)
plt.ylabel("Rzeczywista klasa", fontsize=16)
plt.title("Macierz pomyłek dla modelu Random Forest", fontsize=18)

# Dodanie legendy
plt.text(0.5, -0.50,
         "Legenda:\n"
         "Górny lewy: Prawidłowo zidentyfikowano brak nawrotu (True Negative)\n"
         "Dolny lewy: Nie wykryto nawrotu, który się pojawił (False Negative)\n"
         "Górny prawy: Fałszywy alarm nawrotu, którego nie było (False Positive)\n"
         "Dolny prawy: Prawidłowo zidentyfikowany nawrót (True Positive)\n\n",
         horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, fontsize=8,
         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
plt.tight_layout()
plt.show()

# Analiza waznosci cech, ktore w modelu odgrywaly najwazniejsza role.
# Górne skrzypce dla naszego modelu uczenia graly wielkosc guza, stopien zlosliwosci i wiek
# Co jest obserwacja prawidlowa dla wielu typow nowotworów.
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({"Cecha": feature_names, "Ważność": importances})
feature_importance_df = feature_importance_df.sort_values(by="Ważność", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="Ważność", y="Cecha", data=feature_importance_df, palette="viridis")
plt.title("Ważność cech dla modelu Random Forest")
plt.xlabel("Ważność cechy")
plt.ylabel("Cecha")
plt.show()

#Raport z kroswalidacji
from sklearn.model_selection import cross_val_score, train_test_split

K = 5  # Liczba foldów
cv_scores = cross_val_score(model, X, y, cv=K, scoring='accuracy')
print("Wyniki kroswalidacji (dokładność dla każdego folda):", cv_scores)
print("Średnia dokładność kroswalidacji:", cv_scores.mean())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score, accuracy_score, classification_report, confusion_matrix

# ---- PRUNING LASU LOSOWEGO ----

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score, accuracy_score, classification_report, confusion_matrix

# Używamy recall jako kryterium optymalizacji dla klasy 1 (Nawrót)
recall_scorer = make_scorer(recall_score, pos_label=1)

# Definiujemy przestrzeń hiperparametrów – Tu jest nasz pruning regulujący złożoność drzew
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced', 'balanced_subsample', {0: 1, 1: 8}]
}

# Tworzymy bazowy model Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Używamy GridSearchCV z 5-krotną kroswalidacją, optymalizując Recall dla klasy 1
grid_search = GridSearchCV(rf, param_grid, scoring=recall_scorer, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Najlepsze parametry (pruning):", grid_search.best_params_)

# Używamy najlepszego modelu znalezionego przez GridSearchCV
best_rf = grid_search.best_estimator_
y_pred_pruned = best_rf.predict(X_test)

print("Dokładność modelu po Pruningu:", accuracy_score(y_test, y_pred_pruned))
print("Raport klasyfikacji po pruning:\n", classification_report(y_test, y_pred_pruned))
print("Macierz pomyłek po pruning:\n", confusion_matrix(y_test, y_pred_pruned))

#Raport  poprawia sie wzgledem cechy 1, czyli wykrywania nawrotu, NATOMIAST
#kosztem zarówno precyzji w obrębie cechy 1, jak i precyzji ORAZ czułości
#względem cechy 0 (brak nawrotu).

#Robimy znowu wykresy:

conf_matrix = confusion_matrix(y_test, y_pred_pruned)
plt.figure(figsize=(8,6))
sns.set(font_scale=1.2)
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdYlGn',
                 xticklabels=["Brak nawrotu", "Nawrót"],
                 yticklabels=["Brak nawrotu", "Nawrót"])
plt.xlabel("Przewidziana klasa", fontsize=16)
plt.ylabel("Rzeczywista klasa", fontsize=16)
plt.title("Macierz pomyłek po Pruningu", fontsize=18)

# Dodanie legendy
plt.text(0.5, -0.50,
         "Legenda:\n"
         "Górny lewy: Prawidłowo zidentyfikowano brak nawrotu (True Negative)\n"
         "Dolny lewy: Nie wykryto nawrotu, który się pojawił (False Negative)\n"
         "Górny prawy: Fałszywy alarm nawrotu, którego nie było (False Positive)\n"
         "Dolny prawy: Prawidłowo zidentyfikowany nawrót (True Positive)",
         horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, fontsize=8,
         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

plt.tight_layout()
plt.show()

importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({"Cecha": feature_names, "Ważność": importances})
feature_importance_df = feature_importance_df.sort_values(by="Ważność", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="Ważność", y="Cecha", data=feature_importance_df, palette="viridis")
plt.title("Ważność cech dla modelu Random Forest po Pruningu")
plt.xlabel("Ważność cechy")
plt.ylabel("Cecha")
plt.show()

# Przy analizie waznosci cech po pruningu modelu, ponownie to wielkosc guza, stopien zlosliwosci i wiek
# byly najwazniejsze. Model był testowany również za pomocą Sieci Neuronowej (kod poniżej).
# NATOMIASTraporty nadal były podobne względem modeli Lasu Losowego i Lasu Losowego Po Pruningu.

#-----SIEĆ NEURONOWA-----

#Sieć neuronowa w teorii powinna byc lepsza, ze wzgledu na to iz uczy sie ona wzorcow
# i wykrywa zaleznosci, ktore umykaja lasom losowym. Pozwala na interakcje wielu danym,
# dzieki warstwom neuronalnym, kiedy lasy losowe podejmuja interakcje krok po kroku - jesli X to Y.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


#1) Podział na zbiór treningowy i testowy
X = df.drop("Czy_Nawrot", axis=1)
y = df["Czy_Nawrot"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#2) Standaryzacja cech
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#3a) Budowanie sieci neuronowej - Stara siec, mniej dokladna. Wykomentowana
# model = Sequential([
#     Dense(32, activation='relu', input_shape=(X_train.shape[1],)), #Warstwa wejsciowa
#     Dropout(0.3),  # Dropout dla lepszej generalizacji
#     Dense(16, activation='relu'),  # Warstwa ukryta
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')  # Warstwa wyjściowa (1 neuron, sigmoid)
# ])

#3b) Budowanie sieci neuronowej - Nowa siec, bardziej dokladna.
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

#4) Kompilacja modelu
model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5) Trenowanie modelu
class_weights = {0: 1, 1: 3}
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=2)

#6) Regulacja progu decyzyjnego
y_proba = model.predict(X_test)
# Obniżenie progu do 0.3, by zwiększyć czułość dla Nawrotu
threshold = 0.1
y_pred_adjusted = (y_proba >= threshold).astype("int32")

print("Dokładność Sieci Neuronowej:", accuracy_score(y_test, y_pred))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

#Komentarz: Tutaj można zauważyć, że nasza sieć neuronowa cechuje sie bardzo zmiennymi wynikami.
#Wyniki sa różne przy każdej iteracji. Jest w stanie wyczuć nawrót nowotworu, jednak precyzja i czułość oscylują
#w zakresach 30-70% dla czułości i 50-80% dla precyzji. Nadal nie jest to wynik idealny, ale tutaj
#juz sa lepsze podstawy do dalszej pracy z modelem w celu jego optymalizacji.

#7) Macierz pomyłek - Tak jak w testach na Covid i wszystkich innych Biologicznych sprawach, jest to absolutna konieczność.
# Obliczenie macierzy
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)

# Zastosowanie mapy kolorów "RdYlGn": zielony = dobre przewidywania, czerwony = błędy
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdYlGn', cbar=True,
                 xticklabels=["Brak nawrotu", "Nawrót"],
                 yticklabels=["Brak nawrotu", "Nawrót"])

plt.xlabel("Przewidziana klasa", fontsize=16)
plt.ylabel("Rzeczywista klasa", fontsize=16)
plt.title("Macierz pomyłek dla modelu  Sieci Neuronowe", fontsize=18)

# Dodanie legendy
plt.text(0.5, -0.50,
         "Legenda:\n"
         "Górny lewy: Prawidłowo zidentyfikowano brak nawrotu (True Negative)\n"
         "Dolny lewy: Nie wykryto nawrotu, który się pojawił (False Negative)\n"
         "Górny prawy: Fałszywy alarm nawrotu, którego nie było (False Positive)\n"
         "Dolny prawy: Prawidłowo zidentyfikowany nawrót (True Positive)\n\n",
         horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, fontsize=8,
         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
plt.tight_layout()
plt.show()

#8) Analiza ważności cech (za pomocą SHAP, bo nie możemy wykorzystac tego samego testu jak w przypadku Lasów Losowych)
import shap
# Obliczanie wartości SHAP dla zbioru testowego
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Obliczamy średnią wartość bezwzględną SHAP dla każdej cechy
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
feature_names = X.columns
shap_importance_df = pd.DataFrame({"Cecha": feature_names, "Ważność": mean_abs_shap})
shap_importance_df = shap_importance_df.sort_values(by="Ważność", ascending=False)

# Wizualizacja jako wykres słupkowy, analogicznie do tego z Random Forest
plt.figure(figsize=(10, 5))
sns.barplot(x="Ważność", y="Cecha", data=shap_importance_df, palette="viridis")
plt.title("Ważność cech dla Sieci Neuronowej")
plt.xlabel("Ważność cechy")
plt.ylabel("Cecha")
plt.show()

#Wnioski:
#Z danymi najlepiej radził sobie model Lasów Losowych Po Pruningu, nastepnie Siec Neuronowa, a nastepnie bazowy Las Losowy.
#Dane wykazuja niedoreprezentowanie dla nawrotu nowotworu. W kolejnym hipotetycznym kroku
#użytkownik musiałby przejrzeć dane, uzupełnić lub wprowadzić informacje o nawrotach celem ich
#doreprezentowania, a następnie ponownie uruchomić modele i zobaczyć czy dalej można je optymalizować
#przy jednoczesnym braku spadku parametrow wartosci (recall, precision) dla cechy 0 (brak nawrotu).
#Optymalizacje których dokonałem względem cechy 1 (nawrót nowotworu) silnie oddziaływały na cechę 0 obniżając jej wykrywalność.

