import streamlit as st
import joblib
import numpy as np

#import model
df = joblib.load(open('df.pkl','rb'))
pipe = joblib.load(open('pipe.pkl','rb'))

html_temp = """
<div style ="background-color:#000000;padding:13px;">
    <h1 style ="color:#D3D3D3;text-align:center;">PROJECT UAS AI</h1>
</div>
<div style ="background-color:#D3D3D3; border-top: solid 1px #cca558; margin-bottom: 24px;">
    <h2 style ="color:black;text-align:center;">Armila Suilistyani & Novi Nur Fauziah</h2>
</div>
"""

st.markdown(html_temp, unsafe_allow_html = True)

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
