import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# ===============================
# Konfigurasi Awal
# ===============================
st.set_page_config(page_title="Analisis Sentimen", layout="wide")
st.title("Analisa Sentimen: Perbandingan KNN vs Decision Tree")

# ===============================
# Upload dataset Excel
# ===============================
uploaded_file = st.file_uploader("Unggah Dataset Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Data Awal")
    st.write(df.head())

    text_col = st.selectbox("Pilih kolom teks", df.columns)
    label_col = st.selectbox("Pilih kolom label", df.columns)

    if st.button("Jalankan Analisis"):
        X = df[text_col].astype(str)
        y = df[label_col]

        # Drop missing values
        data = pd.DataFrame({text_col: X, label_col: y}).dropna()
        X = data[text_col]
        y = data[label_col]

        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)
        X_vec = vectorizer.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

        # Model KNN
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        report_knn = classification_report(y_test, y_pred_knn, output_dict=True)

        # Model Decision Tree
        dt = DecisionTreeClassifier(max_depth=10, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        acc_dt = accuracy_score(y_test, y_pred_dt)
        report_dt = classification_report(y_test, y_pred_dt, output_dict=True)

        # Simpan ke session_state
        st.session_state.vectorizer = vectorizer
        st.session_state.knn = knn
        st.session_state.dt = dt
        st.session_state.ready = True

        # Tampilkan Hasil
        st.subheader("Hasil Perbandingan")
        result_df = pd.DataFrame({
            "Model": ["KNN", "Decision Tree"],
            "Accuracy": [acc_knn, acc_dt]
        })
        st.write(result_df)

        # Grafik Akurasi
        fig1, ax1 = plt.subplots()
        sns.barplot(data=result_df, x="Model", y="Accuracy", ax=ax1, palette="Set2")
        ax1.set_ylim(0, 1)
        st.pyplot(fig1)

        # Classification Report
        st.subheader("Classification Report KNN")
        st.write(pd.DataFrame(report_knn).transpose())
        st.subheader("Classification Report Decision Tree")
        st.write(pd.DataFrame(report_dt).transpose())

# ===============================
# Uji Prediksi Manual
# ===============================
st.markdown("---")
st.subheader("Uji Prediksi Teks Baru")
input_text = st.text_area("Masukkan teks untuk diprediksi")

if st.button("Prediksi"):
    if "ready" not in st.session_state or not st.session_state.ready:
        st.warning("Latih model terlebih dahulu (klik 'Jalankan Analisis').")
    elif not input_text.strip():
        st.warning("Teks tidak boleh kosong.")
    else:
        vec = st.session_state.vectorizer.transform([input_text])
        pred_knn = st.session_state.knn.predict(vec)[0]
        pred_dt = st.session_state.dt.predict(vec)[0]
        st.info(f"KNN: **{pred_knn}**")
        st.info(f"Decision Tree: **{pred_dt}**")

# ===============================
# Fungsi Training Model
# ===============================
def train_model(df):
    X = df['Komentar Bersih'].astype(str)
    y = df['Label']
    data = pd.DataFrame({"Komentar Bersih": X, "Label": y}).dropna()
    X = data["Komentar Bersih"]
    y = data["Label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    vektor_x = vectorizer.fit_transform(X)
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(vektor_x, y)
    return vectorizer, model

# ===============================
# Load Model jika ada
# ===============================
try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("decision_tree_model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    vectorizer, model = None, None

# ===============================
# Decision Tree
# ===============================
st.title("Aplikasi Analisis Sentimen Interaktif")
st.write("Menggunakan **Decision Tree** untuk memprediksi sentimen teks.")

# Menu Navigasi
menu = st.sidebar.radio("Menu", ["Prediksi Sentimen", "Latih Model Baru"])

# Halaman Prediksi
if menu == "Prediksi Sentimen":
    if model is None:
        st.warning("Model belum dilatih. Silakan latih model terlebih dahulu di menu 'Latih Model Baru'.")
    else:
        user_input = st.text_area("Masukkan teks atau ulasan:")
        if st.button("Prediksi Sentimen"):
            if user_input.strip() == "":
                st.warning("Harap masukkan teks terlebih dahulu.")
            else:
                input_tfidf = vectorizer.transform([user_input])
                prediction = model.predict(input_tfidf)[0]
                proba = model.predict_proba(input_tfidf)[0]

                # Warna indikator
                if str(prediction).lower() == "positif":
                    st.markdown(f"<h3 style='color:green;'>Sentimen: {prediction}</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:red;'>Sentimen: {prediction}</h3>", unsafe_allow_html=True)

                st.write(f"Confidence: {max(proba)*100:.2f}%")

                # Grafik Confidence Score
                fig, ax = plt.subplots()
                ax.bar(["Negatif", "Positif"], proba, color=["red", "green"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probabilitas")
                ax.set_title("Confidence Score")
                for i, v in enumerate(proba):
                    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
                st.pyplot(fig)

# Halaman Latih Model Baru
elif menu == "Latih Model Baru":
    st.subheader("Unggah Dataset Sentimen (Excel)")
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        if 'Komentar Bersih' not in df.columns or 'Label' not in df.columns:
            st.error("File Excel harus memiliki kolom 'Komentar Bersih' dan 'Label'")
        else:
            st.write("Preview Dataset:", df.head())
            if st.button("Latih Model"):
                vectorizer, model = train_model(df)

                with open("tfidf_vectorizer.pkl", "wb") as f:
                    pickle.dump(vectorizer, f)
                with open("decision_tree_model.pkl", "wb") as f:
                    pickle.dump(model, f)

                st.success("Model dan Vektor berhasil dilatih dan disimpan sebagai 'decision_tree_model.pkl', 'tfidf_vectorizer.pkl'.")

                # Download model
                with open("decision_tree_model.pkl", "rb") as f:
                    st.download_button(
                        label="Download Model",
                        data=f,
                        file_name="decision_tree_model.pkl",
                        mime="application/octet-stream"
                    )

                with open("tfidf_vectorizer.pkl", "rb") as f:
                    st.download_button(
                        label="Download Vektor",
                        data=f,
                        file_name="tfidf_vectorizer.pkl",
                        mime="application/octet-stream"
                    )
    else:
        st.info("Silakan upload dataset Excel terlebih dahulu.")