import io
import os
import tempfile

import numpy as np
import scipy.io.wavfile as wav
import streamlit as st

try:
    from audiorecorder import audiorecorder
except ModuleNotFoundError:
    audiorecorder = None

from pocket_tts import TTSModel


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Voice Over Indonesia",
    page_icon="🎙️",
    layout="centered",
)


# Remove the previous run's temporary reference file. Streamlit reruns this
# script after widget changes, so a session can otherwise accumulate files.
previous_reference = st.session_state.pop("reference_audio_path", None)
if previous_reference and os.path.isfile(previous_reference):
    os.remove(previous_reference)


def save_reference_audio(audio_data, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_data)
        st.session_state.reference_audio_path = temp_audio.name

    return temp_audio.name


# =========================================================
# MODEL
# =========================================================

MODEL_CONFIG = (
    "hf://anak10thn/pocket-tts-indonesian/"
    "indonesian_6l.yaml@"
    "17257664e384561c957b02ac92edd1a24807f0e5"
)


@st.cache_resource
def load_model():
    return TTSModel.load_model(config=MODEL_CONFIG)


with st.spinner("⏳ Memuat model AI Voice Cloning..."):
    try:
        model = load_model()
        st.success(f"✅ Model siap — device: {model.device}")
    except Exception as e:
        st.error("❌ Gagal memuat model.")
        st.exception(e)
        st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("🎙️ AI Voice Over Indonesia")
st.write("Buat voice over Bahasa Indonesia dengan suara referensi Anda.")
st.caption("Gunakan hanya suara milik Anda sendiri atau suara yang Anda punya izin untuk gunakan.")


# =========================================================
# VOICE INPUT
# =========================================================

st.subheader("🎤 1. Masukkan Suara Referensi")

if audiorecorder is None:
    st.warning("⚠️ Rekam dari browser dinonaktifkan karena package `streamlit-audiorecorder` belum terinstal. Silakan upload audio sebagai alternatif.")
    voice_method = "📁 Upload Audio"
else:
    voice_method = st.radio(
        "Pilih sumber suara:",
        ["🎙️ Rekam dari Browser", "📁 Upload Audio"],
        horizontal=True,
    )

reference_audio = None

if voice_method == "🎙️ Rekam dari Browser":
    st.info(
        "Rekam suara yang jelas sekitar 5–15 detik. "
        "Gunakan ruangan yang tenang dan hindari musik/background noise."
    )

    recorded_audio = audiorecorder("🔴 Mulai Rekam", "⏹️ Stop Rekam")

    if len(recorded_audio) > 0:
        sample_width = recorded_audio.sample_width
        channels = recorded_audio.channels
        sample_rate = recorded_audio.frame_rate

        if channels < 1 or sample_rate < 1:
            st.error("Format rekaman tidak valid. Silakan rekam ulang.")
        else:
            samples = np.asarray(recorded_audio.get_array_of_samples())
            sample_scales = {1: 128.0, 2: 32768.0, 3: 8388608.0, 4: 2147483648.0}

            if sample_width not in sample_scales:
                st.error("Format sampel rekaman tidak didukung. Silakan rekam ulang atau upload file WAV.")
            elif samples.size == 0 or samples.size % channels != 0:
                st.error("Data rekaman tidak lengkap. Silakan rekam ulang.")
            else:
                audio_array = samples.astype(np.float32) / sample_scales[sample_width]

                if channels > 1:
                    audio_array = audio_array.reshape(-1, channels).mean(axis=1)

                audio_array = np.clip(audio_array, -1.0, 1.0)
                duration = len(audio_array) / sample_rate

                if duration < 0.5:
                    st.error("Rekaman terlalu singkat. Silakan rekam setidaknya 5 detik.")
                else:
                    peak = float(np.max(np.abs(audio_array)))
                    rms = float(np.sqrt(np.mean(np.square(audio_array))))

                    if duration < 5 or duration > 30:
                        st.warning(f"Durasi rekaman {duration:.1f} detik. Untuk hasil terbaik, gunakan 5–15 detik.")

                    if rms < 0.005:
                        st.warning("Suara rekaman sangat pelan. Coba bicara lebih dekat ke mikrofon.")
                    elif peak > 0.98:
                        st.warning("Suara rekaman mungkin pecah. Kurangi volume atau jauhkan mikrofon.")

                    audio_buffer = io.BytesIO()
                    wav.write(audio_buffer, sample_rate, audio_array)
                    audio_data = audio_buffer.getvalue()
                    reference_audio = save_reference_audio(audio_data, ".wav")

                    st.success(f"✅ Rekaman siap · {duration:.1f} detik")
                    st.audio(audio_data, format="audio/wav")

else:
    uploaded_voice = st.file_uploader(
        "Upload suara referensi",
        type=["wav", "mp3", "m4a", "ogg"],
        help="Gunakan rekaman suara yang jelas tanpa musik atau suara orang lain.",
    )

    if uploaded_voice is not None:
        suffix = os.path.splitext(uploaded_voice.name)[1].lower()
        reference_audio = save_reference_audio(uploaded_voice.getvalue(), suffix)
        st.success(f"✅ Audio diterima: {uploaded_voice.name}")
        st.audio(
            uploaded_voice.getvalue(),
            format={
                ".wav": "audio/wav",
                ".mp3": "audio/mp3",
                ".m4a": "audio/mp4",
                ".ogg": "audio/ogg",
            }.get(suffix, "audio/wav"),
        )


# =========================================================
# SCRIPT
# =========================================================

st.subheader("📝 2. Script Voice Over")

uploaded_script = st.file_uploader(
    "📄 Upload file script (.txt/.md)",
    type=["txt", "md"],
    help="File teks ini akan otomatis dimasukkan ke kolom script di bawah.",
)

script_default = ""
if uploaded_script is not None:
    raw_script = uploaded_script.read()
    try:
        script_default = raw_script.decode("utf-8")
    except UnicodeDecodeError:
        script_default = raw_script.decode("latin-1")

    st.caption(f"File script terupload: {uploaded_script.name}")

text = st.text_area(
    "Masukkan teks yang ingin dibacakan:",
    value=script_default,
    height=220,
    placeholder=(
        "Contoh:\n\n"
        "Halo semuanya!\n"
        "Hari ini saya akan membagikan tips yang sangat menarik untuk kalian."
    ),
)


# =========================================================
# GENERATE
# =========================================================

st.subheader("🚀 3. Generate Voice")

generate = st.button("🎙️ Generate Voice Over", type="primary", use_container_width=True)

if generate:
    if reference_audio is None:
        st.warning("⚠️ Silakan rekam atau upload suara referensi terlebih dahulu.")
        st.stop()

    if not text.strip():
        st.warning("⚠️ Silakan masukkan script terlebih dahulu.")
        st.stop()

    text = text.strip()

    with st.spinner("🎙️ AI sedang membuat voice over dengan referensi suara..."):
        try:
            try:
                voice_state = model.get_state_for_audio_prompt(reference_audio)
                audio = model.generate_audio(voice_state, text)

                audio_np = audio.detach().cpu().numpy().astype(np.float32)

                if audio_np.size == 0 or not np.isfinite(audio_np).all():
                    raise ValueError("Model menghasilkan audio kosong atau tidak valid.")

                max_value = np.max(np.abs(audio_np))
                if max_value > 0:
                    audio_np = audio_np / max_value

                output_buffer = io.BytesIO()
                wav.write(output_buffer, model.sample_rate, audio_np)
                output_data = output_buffer.getvalue()

                duration = len(audio_np) / model.sample_rate

                st.success("✅ Voice Over berhasil dibuat!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⏱️ Durasi", f"{duration:.1f} detik")
                with col2:
                    st.metric("📝 Karakter", f"{len(text):,}")

                st.audio(output_data, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Voice Over",
                    data=output_data,
                    file_name="voice_over_indonesia.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )

            finally:
                temp_reference = st.session_state.pop("reference_audio_path", None)
                if temp_reference and os.path.isfile(temp_reference):
                    os.remove(temp_reference)

        except Exception as e:
            st.error("❌ Gagal membuat voice over.")
            st.exception(e)


# =========================================================
# INFO
# =========================================================

st.markdown("---")

with st.expander("ℹ️ Tips mendapatkan hasil cloning lebih bagus"):
    st.markdown(
        """
**🎤 Rekaman suara**

- Gunakan ruangan yang tenang.
- Jangan menggunakan musik/background sound.
- Bicara dengan volume normal.
- Gunakan microphone yang cukup jelas.
- Rekaman sekitar 5–15 detik sudah cukup untuk percobaan.
- Jangan memasukkan suara orang lain.

**📝 Script**

- Gunakan tanda baca yang jelas.
- Untuk angka, lebih aman tulis dalam bentuk kata.
- Contoh: 15 → lima belas.
- Gunakan kalimat yang tidak terlalu panjang saat testing.
"""
    )