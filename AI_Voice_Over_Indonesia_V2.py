import os
import tempfile

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, sosfiltfilt, resample_poly
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

if audiorecorder is None:
    st.warning("⚠️ Rekam dari browser dinonaktifkan karena package `streamlit-audiorecorder` belum terinstal. Silakan upload audio sebagai alternatif.")
    voice_method = "📁 Upload Audio"
else:
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
        "Rekam suara 8–15 detik dengan suara natural. "
        "Jaga jarak mulut ke mic sekitar 10–20 cm, "
        "ruangan tenang, dan jangan menggunakan musik/background noise."
    )

    recorded_audio = audiorecorder(
        "🔴 Mulai Rekam",
        "⏹️ Stop Rekam"
    )


    # ---------------------------------------------------------
    # AUDIO PREPROCESSING
    # ---------------------------------------------------------

    def convert_recording_to_float32(audio_segment):

        samples = np.array(
            audio_segment.get_array_of_samples()
        )

        sample_width = audio_segment.sample_width

        # Convert raw PCM → float32 [-1, 1]
        if sample_width == 1:

            samples = (
                samples.astype(np.float32) - 128.0
            ) / 128.0

        elif sample_width == 2:

            samples = (
                samples.astype(np.float32)
                / 32768.0
            )

        elif sample_width == 3:

            # 24-bit PCM is represented as signed integer
            samples = (
                samples.astype(np.float32)
                / 8388608.0
            )

        else:

            samples = samples.astype(np.float32)

            peak = np.max(np.abs(samples))

            if peak > 0:
                samples = samples / peak


        # Stereo → mono
        if audio_segment.channels > 1:

            samples = samples.reshape(
                -1,
                audio_segment.channels
            ).mean(axis=1)

        return samples.astype(np.float32)


    def trim_silence(audio, sample_rate):

        if len(audio) == 0:
            return audio

        # Short-term RMS
        frame_length = max(
            int(sample_rate * 0.025),
            256
        )

        hop_length = max(
            int(sample_rate * 0.010),
            128
        )

        rms = []

        for start_idx in range(
            0,
            len(audio) - frame_length + 1,
            hop_length
        ):

            frame = audio[
                start_idx:start_idx + frame_length
            ]

            rms.append(
                np.sqrt(
                    np.mean(frame * frame) + 1e-10
                )
            )

        if not rms:
            return audio

        rms = np.asarray(rms)

        max_rms = np.max(rms)

        if max_rms <= 1e-5:
            return audio

        # Conservative threshold.
        threshold = max(
            max_rms * 0.08,
            1e-4
        )

        active = np.where(
            rms >= threshold
        )[0]

        if len(active) == 0:
            return audio

        first = active[0] * hop_length
        last = min(
            active[-1] * hop_length + frame_length,
            len(audio)
        )

        # Keep a little natural room around speech.
        padding = int(sample_rate * 0.08)

        first = max(0, first - padding)
        last = min(len(audio), last + padding)

        return audio[first:last]


    def remove_low_frequency_noise(audio, sample_rate):

        # Remove DC / rumble below ~70 Hz.
        if sample_rate <= 160:
            return audio

        highcut = min(
            70.0,
            sample_rate * 0.20
        )

        sos = butter(
            4,
            highcut,
            btype="highpass",
            fs=sample_rate,
            output="sos"
        )

        try:
            return sosfiltfilt(sos, audio).astype(
                np.float32
            )
        except ValueError:
            return audio


    def reduce_background_noise(audio, sample_rate):

        """
        Conservative spectral noise reduction.

        The first 0.20–0.35 sec is treated as a possible
        noise-only region. Reduction is intentionally mild
        so the cloned voice does not become metallic.
        """

        if len(audio) < sample_rate * 0.5:
            return audio

        n_fft = 1024

        # Keep FFT size valid for short recordings.
        while n_fft > len(audio):
            n_fft //= 2

        if n_fft < 256:
            return audio

        hop = n_fft // 4
        window = np.hanning(n_fft)

        # Estimate noise from the beginning of the recording.
        noise_duration = min(
            0.35,
            max(0.20, len(audio) / sample_rate * 0.20)
        )

        noise_samples = max(
            n_fft,
            int(sample_rate * noise_duration)
        )

        noise_samples = min(
            noise_samples,
            len(audio)
        )

        noise_part = audio[:noise_samples]

        if np.sqrt(
            np.mean(noise_part * noise_part) + 1e-10
        ) > 0.08:
            # Beginning may already contain speech.
            # Avoid aggressive subtraction.
            return audio

        # STFT using scipy.signal.stft would be more verbose;
        # use frame-based FFT for a small, conservative gate.
        frames = []
        positions = range(
            0,
            max(1, len(audio) - n_fft + 1),
            hop
        )

        for pos in positions:
            frame = audio[pos:pos + n_fft]

            if len(frame) < n_fft:
                frame = np.pad(
                    frame,
                    (0, n_fft - len(frame))
                )

            spectrum = np.fft.rfft(
                frame * window
            )

            frames.append(spectrum)

        if not frames:
            return audio

        spectra = np.asarray(frames)

        noise_frame_count = max(
            1,
            int(noise_samples / hop)
        )

        noise_mag = np.median(
            np.abs(
                spectra[:noise_frame_count]
            ),
            axis=0
        )

        output = np.zeros_like(
            audio,
            dtype=np.float32
        )

        weight = np.zeros_like(
            audio,
            dtype=np.float32
        )

        for frame_idx, pos in enumerate(positions):

            frame = audio[pos:pos + n_fft]

            if len(frame) < n_fft:
                frame = np.pad(
                    frame,
                    (0, n_fft - len(frame))
                )

            spectrum = spectra[frame_idx]

            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Mild spectral subtraction.
            cleaned_mag = np.maximum(
                magnitude - noise_mag * 0.55,
                magnitude * 0.35
            )

            cleaned = np.fft.irfft(
                cleaned_mag * np.exp(1j * phase),
                n=n_fft
            ).astype(np.float32)

            end_pos = min(
                pos + n_fft,
                len(audio)
            )

            length = end_pos - pos

            output[pos:end_pos] += (
                cleaned[:length] * window[:length]
            )

            weight[pos:end_pos] += (
                window[:length] ** 2
            )

        valid = weight > 1e-6

        result = audio.copy()

        result[valid] = (
            output[valid] / weight[valid]
        )

        return result.astype(np.float32)


    def normalize_audio(audio):

        if len(audio) == 0:
            return audio

        # Remove DC offset.
        audio = audio - np.mean(audio)

        peak = np.max(np.abs(audio))

        if peak <= 1e-8:
            return audio.astype(np.float32)

        # Target peak ≈ -1 dBFS.
        target_peak = 10 ** (-1 / 20)

        audio = (
            audio / peak
        ) * target_peak

        return audio.astype(np.float32)


    def preprocess_reference_audio(
        audio_segment
    ):

        sample_rate = audio_segment.frame_rate

        audio = convert_recording_to_float32(
            audio_segment
        )

        # 1. Trim excessive silence.
        audio = trim_silence(
            audio,
            sample_rate
        )

        # 2. Remove low-frequency rumble / DC.
        audio = remove_low_frequency_noise(
            audio,
            sample_rate
        )

        # 3. Conservative background-noise reduction.
        audio = reduce_background_noise(
            audio,
            sample_rate
        )

        # 4. Final silence trim.
        audio = trim_silence(
            audio,
            sample_rate
        )

        # 5. Normalize.
        audio = normalize_audio(audio)

        return audio, sample_rate


    if len(recorded_audio) > 0:

        with st.spinner(
            "🧹 Membersihkan dan menyiapkan rekaman..."
        ):

            try:

                # Process recording.
                audio_array, sample_rate = (
                    preprocess_reference_audio(
                        recorded_audio
                    )
                )

                # Basic quality check.
                duration = (
                    len(audio_array)
                    / sample_rate
                )

                rms = np.sqrt(
                    np.mean(
                        audio_array * audio_array
                    ) + 1e-10
                )

                peak = np.max(
                    np.abs(audio_array)
                )

                clipping_ratio = np.mean(
                    np.abs(audio_array) >= 0.99
                )

                # Save processed reference audio.
                temp_voice = (
                    tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav"
                    )
                )

                reference_audio = (
                    temp_voice.name
                )

                temp_voice.close()

                wav.write(
                    reference_audio,
                    sample_rate,
                    audio_array
                )

                # -------------------------------------------------
                # QUALITY FEEDBACK
                # -------------------------------------------------

                if duration < 5:

                    st.warning(
                        "⚠️ Rekaman terlalu pendek. "
                        "Gunakan sekitar 8–15 detik untuk hasil "
                        "voice cloning yang lebih konsisten."
                    )

                elif duration > 30:

                    st.warning(
                        "⚠️ Rekaman cukup panjang. "
                        "Untuk voice cloning, 8–15 detik suara "
                        "bersih biasanya lebih praktis."
                    )

                else:

                    st.success(
                        "✅ Rekaman berhasil dibersihkan."
                    )

                quality_col1, quality_col2, quality_col3 = (
                    st.columns(3)
                )

                with quality_col1:

                    st.metric(
                        "Durasi",
                        f"{duration:.1f} detik"
                    )

                with quality_col2:

                    st.metric(
                        "RMS",
                        f"{rms:.3f}"
                    )

                with quality_col3:

                    if clipping_ratio < 0.001:
                        clipping_status = "✅ Aman"
                    else:
                        clipping_status = "⚠️ Clipping"

                    st.metric(
                        "Peak",
                        clipping_status
                    )

                if clipping_ratio >= 0.001:

                    st.warning(
                        "⚠️ Rekaman mengalami clipping. "
                        "Turunkan volume mic atau jauhkan mulut "
                        "sedikit dari microphone lalu rekam ulang."
                    )

                st.audio(
                    reference_audio,
                    format="audio/wav"
                )

                st.caption(
                    "Audio di atas adalah versi yang sudah "
                    "dibersihkan dan akan digunakan sebagai "
                    "referensi voice cloning."
                )

            except Exception as e:

                st.error(
                    "❌ Gagal memproses rekaman."
                )

                st.exception(e)

                reference_audio = None


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

