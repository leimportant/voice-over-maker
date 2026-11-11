import io
import re

import numpy as np
import scipy.io.wavfile as wav
import streamlit as st
import torch
from transformers import AutoTokenizer, VitsModel


MODEL_NAME = "facebook/mms-tts-ind"
MAX_CHUNK_CHARS = 300

st.set_page_config(
    page_title="AI Voice Over Indonesia",
    page_icon="🎙️",
    layout="centered"
)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = VitsModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def split_long_sentence(sentence, max_chars):
    words = sentence.split()
    chunks = []
    current = ""

    for word in words:
        while len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(word[:max_chars])
            word = word[max_chars:]

        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = word
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def split_text(text, max_chars=MAX_CHUNK_CHARS):
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_sentence(sentence, max_chars))
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def change_speed(audio, speed):
    if speed == 1.0 or audio.size < 2:
        return audio

    new_length = max(1, int(len(audio) / speed))
    old_positions = np.arange(len(audio))
    new_positions = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def generate_audio(text, speed, pause, tokenizer, model, device):
    chunks = split_text(text)
    sample_rate = model.config.sampling_rate
    pause_audio = np.zeros(int(sample_rate * pause), dtype=np.float32)
    generated_chunks = []
    progress = st.progress(0, text="Menyiapkan audio...")

    for index, chunk in enumerate(chunks):
        inputs = tokenizer(chunk, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            waveform = model(**inputs).waveform

        audio = waveform.squeeze().cpu().numpy().astype(np.float32)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Model menghasilkan audio kosong atau tidak valid.")

        generated_chunks.append(change_speed(audio, speed))
        if pause > 0 and index < len(chunks) - 1:
            generated_chunks.append(pause_audio)

        progress.progress(
            (index + 1) / len(chunks),
            text=f"Membuat audio {index + 1}/{len(chunks)}..."
        )

    progress.empty()
    return np.concatenate(generated_chunks), sample_rate


st.title("🎙️ AI Voice Over Indonesia")
st.write(
    "Ubah script Bahasa Indonesia menjadi voice over. "
    "Teks panjang diproses per bagian agar lebih stabil."
)

try:
    with st.spinner("Memuat AI Voice Model..."):
        tokenizer, model, device = load_model()
except Exception as error:
    st.error("Gagal memuat model suara. Periksa koneksi dan dependensi, lalu coba lagi.")
    st.exception(error)
    st.stop()

if device == "cuda":
    st.success("GPU terdeteksi — generasi suara menggunakan CUDA.")
else:
    st.info("Menggunakan CPU untuk generasi suara.")

text = st.text_area(
    "📝 Script voice over",
    placeholder="Tulis teks yang ingin dibacakan...",
    height=240,
    max_chars=10000
)
st.caption(f"{len(text):,} / 10.000 karakter")

speed_col, pause_col = st.columns(2)
with speed_col:
    speed = st.slider(
        "⚡ Kecepatan",
        min_value=0.7,
        max_value=1.5,
        value=1.0,
        step=0.1,
        help="Perubahan kecepatan melalui resampling juga dapat mengubah pitch."
    )
with pause_col:
    pause = st.slider(
        "⏸️ Jeda antar bagian (detik)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

if st.button(
    "🎙️ Generate Voice Over",
    type="primary",
    use_container_width=True
):
    if not text.strip():
        st.warning("Silakan masukkan script terlebih dahulu.")
    else:
        try:
            with st.spinner("Sedang membuat voice over..."):
                audio, sample_rate = generate_audio(
                    text,
                    speed,
                    pause,
                    tokenizer,
                    model,
                    device
                )

                peak = float(np.max(np.abs(audio)))
                if peak == 0:
                    raise ValueError("Audio yang dihasilkan tidak memiliki suara.")

                audio_int16 = np.rint(
                    np.clip(audio / peak, -1.0, 1.0) * 32767
                ).astype(np.int16)
                output_buffer = io.BytesIO()
                wav.write(output_buffer, sample_rate, audio_int16)
                output_data = output_buffer.getvalue()

            duration = len(audio_int16) / sample_rate
            st.success("Voice over berhasil dibuat!")
            st.metric("⏱️ Durasi audio", f"{duration:.1f} detik")
            st.audio(output_data, format="audio/wav")
            st.download_button(
                "⬇️ Download Audio",
                data=output_data,
                file_name="voice_over_indonesia.wav",
                mime="audio/wav",
                use_container_width=True
            )
        except Exception as error:
            st.error("Gagal membuat voice over.")
            st.exception(error)

st.markdown("---")
st.caption("AI Voice Over Indonesia · Powered by MMS TTS")
