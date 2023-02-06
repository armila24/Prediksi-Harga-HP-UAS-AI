import streamlit as st
import pickle
import numpy as np

#import model
df = pickle.load(open('df.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'))

st.title("Prediksi Harga Handphone")

#brand hp
merk = st.selectbox('Brand',df['MERK'].unique())

#tipe hp
tipe = st.selectbox('Tipe',df['TIPE'].unique())

#pilihan ram
ram = st.selectbox('RAM(GB)',df['RAM'].unique())

#pilihan memori
memori = st.selectbox('MEMORI(GB)',df['MEMORI'].unique())

query = np.array([merk,tipe,ram,memori])
query = query.reshape(1,4)
st.title("PREDIKSI HARGA HANDPHONE (dalam Juta): Rp "+str(int(np.exp(pipe.predict(query)[0]))))