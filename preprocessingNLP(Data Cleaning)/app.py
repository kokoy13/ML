import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')

# 1. Baca file Excel
df = pd.read_excel("data_scrapping.xlsx")
df.head() #menampilkan 5 data teratas

#2
df['Komentar Bersih'] = ''

# 3. Siapkan alat NLP
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# 4. Preprocessing satu per satu baris (tanpa fungsi)
for i in range(len(df)):
    text = str(df.loc[i, 'Komentar'])

    # Skip jika teks kosong atau NaN
    if text.strip() == '' or text.lower() == 'nan':
        continue

    # Lowercase
    text = text.lower()
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenisasi
    tokens = word_tokenize(text)
    # Hapus stopword
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    # Gabungkan kembali
    clean_text = ' '.join(tokens)
    # Simpan ke kolom baru
    df.loc[i, 'Komentar Bersih'] = clean_text
    
# Hapus baris yang kosong di kolom baru
df = df[df['Komentar Bersih'] != '']

# 5. Simpan hasil
df.to_excel("data_toothless_clean.xlsx", index=False)

print(" Pre-processing selesai, hasil disimpan ke 'data_toothless_clean.xlsx'")