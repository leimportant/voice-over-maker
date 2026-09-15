import streamlit as st
import torch
import scipy.io.wavfile as wav
from transformers import VitsModel, AutoTokenizer
import os

st.set_page_config(
    page_title="AI Voice Over Indonesia",
    page_icon="🎙️"
)

st.title("🎙️ AI Voice Over Indonesia")
st.write("Ubah teks menjadi suara bahasa Indonesia.")

@st.cache_resource
def load_model():
    model_name = "facebook/mms-tts-ind"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return tokenizer, model, device


tokenizer, model, device = load_model()

text = st.text_area(
    "📝 Teks",
    placeholder="Tulis teks yang ingin dibacakan...",
    height=200
)

speed = st.selectbox(
    "⚡ Kecepatan Suara",
    [0.5, 1.0, 1.5, 2.0],
    index=1
)

if st.button("🎙️ Generate Voice", type="primary"):

    if not text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:

        with st.spinner("Sedang membuat suara..."):

            inputs = tokenizer(
                text,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                output = model(**inputs).waveform

            audio = output.squeeze().cpu().numpy()

            sample_rate = model.config.sampling_rate

            output_file = "/tmp/voice_over_indonesia.wav"

            wav.write(
                output_file,
                sample_rate,
                audio
            )

        st.success("Voice Over berhasil dibuat!")

        st.audio(output_file, format="audio/wav")

        with open(output_file, "rb") as file:
            st.download_button(
                label="⬇️ Download Audio",
                data=file,
                file_name="voice_over_indonesia.wav",
                mime="audio/wav"
            )

st.markdown("---")
st.caption("AI Voice Over Indonesia · Powered by MMS TTS")
