import streamlit as st
import torch
import scipy.io.wavfile as wav
import numpy as np
import os
import re
import tempfile

from transformers import VitsModel, AutoTokenizer


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Voice Over Indonesia",
    page_icon="🎙️",
    layout="centered"
)

MODEL_NAME = "facebook/mms-tts-ind"


# =========================================================
# HEADER
# =========================================================

st.title("🎙️ AI Voice Over Indonesia")
st.write(
    "Ubah script menjadi suara bahasa Indonesia "
    "untuk kebutuhan TikTok, Reels, YouTube, dan konten lainnya."
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = VitsModel.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    model.eval()

    return tokenizer, model, device


with st.spinner("Memuat AI Voice Model..."):
    tokenizer, model, device = load_model()


# =========================================================
# DEVICE INFO
# =========================================================

if device == "cuda":
    st.success("🚀 GPU terdeteksi — Voice generation menggunakan CUDA.")
else:
    st.info(
        "💻 GPU tidak terdeteksi — Voice generation menggunakan CPU."
    )


# =========================================================
# CONTENT TYPE
# =========================================================

content_type = st.selectbox(
    "🎬 Jenis Konten",
    [
        "TikTok / Reels",
        "YouTube",
        "Storytelling",
        "Tutorial",
        "Iklan / Promosi",
        "Podcast",
        "Umum"
    ]
)


# =========================================================
# TEXT INPUT
# =========================================================

default_placeholder = {
    "TikTok / Reels":
        "Contoh:\nTahukah kamu? Ada satu fakta menarik yang jarang diketahui...",

    "YouTube":
        "Tulis script YouTube kamu di sini...",

    "Storytelling":
        "Tulis cerita yang ingin dibacakan...",

    "Tutorial":
        "Langkah pertama, buka aplikasi...\n\nLangkah kedua...",

    "Iklan / Promosi":
        "Punya masalah dengan produk ini?\nSekarang ada solusinya...",

    "Podcast":
        "Halo semuanya, selamat datang kembali di podcast kita...",

    "Umum":
        "Tulis teks yang ingin dibacakan..."
}


text = st.text_area(
    "📝 Script Voice Over",
    placeholder=default_placeholder[content_type],
    height=300
)


# =========================================================
# SETTINGS
# =========================================================

st.subheader("⚙️ Pengaturan Suara")

col1, col2 = st.columns(2)

with col1:

    speed = st.slider(
        "⚡ Kecepatan",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

with col2:

    pause = st.slider(
        "⏸️ Jeda antar kalimat",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05
    )


# =========================================================
# TEXT PROCESSING
# =========================================================

def split_text(text, max_chars=350):

    """
    Membagi teks panjang menjadi beberapa bagian
    berdasarkan kalimat.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current = ""

    for sentence in sentences:

        if len(current) + len(sentence) + 1 <= max_chars:

            if current:
                current += " " + sentence
            else:
                current = sentence

        else:

            if current:
                chunks.append(current)

            current = sentence

    if current:
        chunks.append(current)

    return chunks


# =========================================================
# SPEED PROCESSING
# =========================================================

def change_speed(audio, speed):

    """
    Mengubah playback speed menggunakan resampling.

    speed > 1  = lebih cepat
    speed < 1  = lebih lambat
    """

    if speed == 1.0:
        return audio

    original_length = len(audio)

    new_length = int(original_length / speed)

    old_indices = np.linspace(
        0,
        original_length - 1,
        original_length
    )

    new_indices = np.linspace(
        0,
        original_length - 1,
        new_length
    )

    audio = np.interp(
        new_indices,
        old_indices,
        audio
    )

    return audio.astype(np.float32)


# =========================================================
# GENERATE AUDIO
# =========================================================

def generate_audio(text):

    chunks = split_text(text)

    generated_audio = []

    sample_rate = model.config.sampling_rate

    pause_samples = int(
        sample_rate * pause
    )

    silence = np.zeros(
        pause_samples,
        dtype=np.float32
    )

    progress = st.progress(0)

    status = st.empty()

    for index, chunk in enumerate(chunks):

        status.text(
            f"Membuat voice {index + 1}/{len(chunks)}..."
        )

        inputs = tokenizer(
            chunk,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = model(
                **inputs
            ).waveform

        audio = (
            output
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Apply speed
        audio = change_speed(
            audio,
            speed
        )

        generated_audio.append(audio)

        # Add pause between chunks
        if index < len(chunks) - 1:
            generated_audio.append(silence)

        progress.progress(
            (index + 1) / len(chunks)
        )

    status.empty()
    progress.empty()

    final_audio = np.concatenate(
        generated_audio
    )

    return final_audio, sample_rate


# =========================================================
# GENERATE BUTTON
# =========================================================

if st.button(
    "🎙️ Generate Voice Over",
    type="primary",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "⚠️ Silakan masukkan script terlebih dahulu."
        )

    else:

        with st.spinner(
            "🎙️ Sedang membuat voice over..."
        ):

            try:

                audio, sample_rate = generate_audio(
                    text
                )

                # Normalize audio
                max_value = np.max(
                    np.abs(audio)
                )

                if max_value > 0:
                    audio = audio / max_value

                # Convert to int16
                audio_int16 = (
                    audio * 32767
                ).astype(np.int16)

                # Temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                )

                output_file = temp_file.name

                temp_file.close()

                wav.write(
                    output_file,
                    sample_rate,
                    audio_int16
                )

                # =================================================
                # DURATION
                # =================================================

                duration = len(audio) / sample_rate

                minutes = int(duration // 60)
                seconds = int(duration % 60)

                st.success(
                    "✅ Voice Over berhasil dibuat!"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "⏱️ Durasi",
                        f"{minutes}:{seconds:02d}"
                    )

                with col2:
                    st.metric(
                        "📝 Karakter",
                        f"{len(text):,}"
                    )

                with col3:
                    st.metric(
                        "⚡ Speed",
                        f"{speed:.1f}x"
                    )

                # =================================================
                # AUDIO PLAYER
                # =================================================

                st.audio(
                    output_file,
                    format="audio/wav"
                )

                # =================================================
                # DOWNLOAD
                # =================================================

                with open(
                    output_file,
                    "rb"
                ) as file:

                    st.download_button(
                        label="⬇️ Download Voice Over",
                        data=file,
                        file_name=(
                            "voice_over_indonesia.wav"
                        ),
                        mime="audio/wav",
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    f"❌ Terjadi error: {str(e)}"
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎙️ AI Voice Over Indonesia · "
    "Powered by Facebook MMS TTS"
)

