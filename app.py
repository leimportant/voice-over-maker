import os
import tempfile

import numpy as np
import scipy.io.wavfile as wav
import streamlit as st

from audiorecorder import audiorecorder
from pocket_tts import TTSModel


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Voice Over Indonesia",
    page_icon="🎙️",
    layout="centered"
)


# =========================================================
# HEADER
# =========================================================

st.title("🎙️ AI Voice Over Indonesia")

st.write(
    "Buat voice over Bahasa Indonesia menggunakan "
    "suara Anda sendiri."
)

st.caption(
    "Gunakan hanya suara milik Anda sendiri atau suara "
    "yang Anda memiliki izin untuk gunakan."
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_CONFIG = (
    "hf://anak10thn/pocket-tts-indonesian/"
    "indonesian_6l.yaml@"
    "17257664e384561c957b02ac92edd1a24807f0e5"
)


@st.cache_resource
def load_model():

    model = TTSModel.load_model(
        config=MODEL_CONFIG
    )

    return model


with st.spinner(
    "⏳ Memuat model AI Voice Cloning..."
):

    try:

        model = load_model()

        st.success(
            f"✅ Model siap — device: {model.device}"
        )

    except Exception as e:

        st.error(
            "❌ Gagal memuat model."
        )

        st.exception(e)

        st.stop()


# =========================================================
# VOICE INPUT
# =========================================================

st.subheader("🎤 1. Masukkan Suara Anda")

voice_method = st.radio(
    "Pilih sumber suara:",
    [
        "🎙️ Rekam dari Browser",
        "📁 Upload Audio"
    ],
    horizontal=True
)


reference_audio = None


# =========================================================
# RECORD FROM BROWSER
# =========================================================

if voice_method == "🎙️ Rekam dari Browser":

    st.info(
        "Rekam suara yang jelas sekitar 5–15 detik. "
        "Gunakan ruangan yang tenang dan hindari musik/background noise."
    )

    recorded_audio = audiorecorder(
        "🔴 Mulai Rekam",
        "⏹️ Stop Rekam"
    )

    if len(recorded_audio) > 0:

        # Convert stereo/mono data
        audio_array = np.array(
            recorded_audio.get_array_of_samples()
        )

        sample_width = recorded_audio.sample_width

        if sample_width == 2:

            audio_array = (
                audio_array.astype(np.float32)
                / 32768.0
            )

        elif sample_width == 1:

            audio_array = (
                audio_array.astype(np.float32)
                - 128
            ) / 128.0

        else:

            audio_array = audio_array.astype(
                np.float32
            )

        # Convert stereo → mono
        if recorded_audio.channels > 1:

            audio_array = audio_array.reshape(
                -1,
                recorded_audio.channels
            ).mean(axis=1)

        # Save temporary WAV
        temp_voice = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        reference_audio = temp_voice.name

        temp_voice.close()

        wav.write(
            reference_audio,
            recorded_audio.frame_rate,
            audio_array
        )

        st.success(
            "✅ Rekaman berhasil."
        )

        st.audio(
            reference_audio,
            format="audio/wav"
        )


# =========================================================
# UPLOAD AUDIO
# =========================================================

else:

    uploaded_voice = st.file_uploader(
        "Upload suara referensi",
        type=[
            "wav",
            "mp3",
            "m4a",
            "ogg"
        ],
        help=(
            "Gunakan rekaman suara yang jelas "
            "tanpa musik atau suara orang lain."
        )
    )

    if uploaded_voice is not None:

        suffix = os.path.splitext(
            uploaded_voice.name
        )[1]

        temp_voice = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        temp_voice.write(
            uploaded_voice.getbuffer()
        )

        temp_voice.close()

        reference_audio = temp_voice.name

        st.success(
            f"✅ Audio diterima: {uploaded_voice.name}"
        )

        st.audio(
            uploaded_voice
        )


# =========================================================
# SCRIPT
# =========================================================

st.subheader("📝 2. Script Voice Over")

text = st.text_area(
    "Masukkan teks yang ingin dibacakan:",
    height=220,
    placeholder=(
        "Contoh:\n\n"
        "Halo semuanya!\n"
        "Hari ini saya akan membagikan tips "
        "yang sangat menarik untuk kalian."
    )
)


# =========================================================
# SETTINGS
# =========================================================

st.subheader("⚙️ 3. Pengaturan")

col1, col2 = st.columns(2)

with col1:

    temperature = st.slider(
        "🎛️ Temperature",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help=(
            "Nilai lebih rendah biasanya menghasilkan "
            "suara yang lebih konsisten."
        )
    )


with col2:

    eos_threshold = st.slider(
        "⏱️ EOS Threshold",
        min_value=-8.0,
        max_value=-3.0,
        value=-6.0,
        step=0.5,
        help=(
            "Nilai yang lebih rendah membantu "
            "teks panjang tidak terpotong."
        )
    )


# =========================================================
# GENERATE
# =========================================================

st.subheader("🚀 4. Generate Voice")

generate = st.button(
    "🎙️ Generate Voice Over",
    type="primary",
    use_container_width=True
)


if generate:

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if reference_audio is None:

        st.warning(
            "⚠️ Silakan rekam atau upload suara Anda terlebih dahulu."
        )

        st.stop()


    if not text.strip():

        st.warning(
            "⚠️ Silakan masukkan script terlebih dahulu."
        )

        st.stop()


    # -----------------------------------------------------
    # CLEAN TEXT
    # -----------------------------------------------------

    text = text.strip()

    # Model ini lebih cocok jika angka ditulis sebagai kata.
    # Contoh: 15 -> lima belas
    #
    # Kita tidak melakukan konversi otomatis di sini
    # agar script asli pengguna tidak berubah.


    # -----------------------------------------------------
    # GENERATE
    # -----------------------------------------------------

    with st.spinner(
        "🎙️ AI sedang membuat voice over..."
    ):

        try:

            # Get voice characteristics
            voice_state = (
                model.get_state_for_audio_prompt(
                    reference_audio
                )
            )

            # Generate
            audio = model.generate_audio(
                voice_state,
                text
            )

            # Convert tensor → numpy
            audio_np = (
                audio
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # -------------------------------------------------
            # NORMALIZE
            # -------------------------------------------------

            max_value = np.max(
                np.abs(audio_np)
            )

            if max_value > 0:

                audio_np = (
                    audio_np / max_value
                )

            # -------------------------------------------------
            # SAVE
            # -------------------------------------------------

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )

            output_path = output_file.name

            output_file.close()

            wav.write(
                output_path,
                model.sample_rate,
                audio_np
            )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            duration = (
                len(audio_np)
                / model.sample_rate
            )

            st.success(
                "✅ Voice Over berhasil dibuat!"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "⏱️ Durasi",
                    f"{duration:.1f} detik"
                )

            with col2:

                st.metric(
                    "📝 Karakter",
                    f"{len(text):,}"
                )

            st.audio(
                output_path,
                format="audio/wav"
            )

            # -------------------------------------------------
            # DOWNLOAD
            # -------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as audio_file:

                st.download_button(
                    label="⬇️ Download Voice Over",
                    data=audio_file.read(),
                    file_name=(
                        "voice_over_indonesia.wav"
                    ),
                    mime="audio/wav",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "❌ Gagal membuat voice over."
            )

            st.exception(e)


# =========================================================
# INFO
# =========================================================

st.markdown("---")

with st.expander(
    "ℹ️ Tips mendapatkan hasil cloning lebih bagus"
):

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
- Contoh: `15` → `lima belas`.
- Gunakan kalimat yang tidak terlalu panjang saat testing.

**⚡ Performance**

Model 6-layer relatif ringan dan dirancang agar dapat berjalan di CPU.
Untuk produksi dengan banyak request, GPU akan tetap lebih nyaman.
"""
    )


st.caption(
    "AI Voice Over Indonesia · Pocket TTS Indonesian"
)

