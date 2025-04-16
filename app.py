import streamlit as st
import functions
import numpy as np
import random
import os
import base64

st.title("Lo-Fi Music Generator")
st.sidebar.header("Effects Options")
tape_comp = st.sidebar.checkbox("Apply Tape Compression")
tape_intensity = st.sidebar.slider("Tape Compression Intensity", min_value=0.1, max_value=2.0, value=0.5, step=0.1) if tape_comp else 0.5
apply_artifact = st.sidebar.checkbox("Apply Lofi Artifact")
artifact_intensity = st.sidebar.slider("Artifact Intensity", min_value=0.0, max_value=0.05, value=0.02, step=0.005) if apply_artifact else 0.02
artifact_type = st.sidebar.selectbox("Artifact Type", ["crackle", "fire"]) if apply_artifact else "crackle"
st.sidebar.header("Drum Options")
drum_choice = st.sidebar.selectbox("Choose Drum Track", ["Random", "Drum Pattern 1", "Drum Pattern 2", "Drum Pattern 3", "Drum Pattern 4", "Drum Pattern 5"])
st.sidebar.header("Transpose Options")
transpose_option = st.sidebar.radio("Transpose", ["Fixed", "Random"])
if transpose_option == "Fixed":
    transpose_value = st.sidebar.number_input("Transpose Offset (-5 to 5)", min_value=-5, max_value=5, value=0, step=1)
    randomize_transpose = False
else:
    randomize_transpose = True
    transpose_value = 0
st.sidebar.header("Music Length")
total_seconds = st.sidebar.slider("Total Music Length (seconds)", min_value=functions.BEAT_TOTAL, max_value=64, value=16, step=functions.BEAT_TOTAL)
num_measures = int(total_seconds / functions.BEAT_TOTAL)

if st.button("Generate Music"):
    with st.spinner("Generating..."):
        audio, sr = functions.generate_music(tape_compression=tape_comp,
                                              tape_intensity=tape_intensity,
                                              apply_artifact=apply_artifact,
                                              artifact_intensity=artifact_intensity,
                                              artifact_type=artifact_type,
                                              drum_choice=drum_choice,
                                              randomize_transpose=randomize_transpose,
                                              transpose_value=transpose_value,
                                              randomness=True,
                                              num_measures=num_measures)
    st.success("Music Generated!")
    cat_source = random.choice(["app/dancing_cat.gif",'app/chill_cat.gif'])
    st.image(cat_source, use_column_width=True)
    wav_data = functions.wav_bytes(audio, sr)
    audio_base64 = base64.b64encode(wav_data).decode()
    audio_html = f'<audio controls autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)
    st.download_button("Download WAV", data=wav_data, file_name="generated_music.wav", mime="audio/wav")
