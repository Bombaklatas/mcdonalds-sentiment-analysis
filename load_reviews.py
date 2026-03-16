import pandas as pd
import sys
import re

# Nustatomas UTF-8 encoding konsolės išvedimui
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Įkeliamas CSV failas su encoding parametru
try:
    df = pd.read_csv('McDonald_s_Reviews.csv', encoding='utf-8')
except UnicodeDecodeError:
    # Jei UTF-8 nepavyksta, bandoma su kitomis koduotėmis
    try:
        df = pd.read_csv('McDonald_s_Reviews.csv', encoding='latin-1')
    except:
        df = pd.read_csv('McDonald_s_Reviews.csv', encoding='cp1252', errors='ignore')

print(f"Pradinis duomenų kiekis: {len(df)} eilucių")

# 1. Pašalinamos tuščios eilutės
df = df.dropna(subset=['review'])  # Pašalinamos eilutės, kur review yra NaN
df = df[df['review'].str.strip() != '']  # Pašalinamos eilutės su tuščiu tekstu
print(f"Po tuščių eilučių pašalinimo: {len(df)} eilucių")

# 2. Išvalomas tekstas nuo skyrybos ženklų (paliekamos tik raidės, skaičiai ir tarpai)
df['review_cleaned'] = df['review'].astype(str).apply(
    lambda x: re.sub(r'[^\w\s]', '', x)
)

# 3. Sukuriamas sentiment stulpelis
# Išgaunamas skaičius iš rating stulpelio (pvz., "4 stars" -> 4)
df['rating_num'] = df['rating'].astype(str).str.extract(r'(\d+)').astype(float)

# Sentiment: 4-5 žvaigždutės = 1, 1-2 žvaigždutės = 0
df['sentiment'] = df['rating_num'].apply(
    lambda x: 1 if x >= 4 else (0 if x <= 2 else None)
)

# Pašalinamos eilutės, kur rating yra 3 (neutralus)
df = df.dropna(subset=['sentiment'])

print(f"Po neutralių (3 žvaigždutės) pašalinimo: {len(df)} eilucių")
print(f"\nSentiment pasiskirstymas:")
print(df['sentiment'].value_counts())

# Rodomos pirmos 5 eilutės su paruoštais duomenimis
print("\nPirmos 5 eilutes su paruoštais duomenimis:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
print(df[['review', 'review_cleaned', 'rating', 'rating_num', 'sentiment']].head(5))

# Išsaugomas paruoštas duomenų rinkinys
df.to_csv('McDonald_s_Reviews_prepared.csv', index=False, encoding='utf-8')
print("\nParuošti duomenys išsaugoti į 'McDonald_s_Reviews_prepared.csv'")
