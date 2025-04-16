# tests/test_functions.py
import pytest
import numpy as np
import functions  # Import the module we're testing
import wave
import io

# --- Constants for Testing ---
TEST_SAMPLE_RATE = functions.SAMPLE_RATE # Use the sample rate from the module
TEST_DURATION_SECONDS = 2.0 # Short duration for faster tests
TEST_NUM_MEASURES = int(TEST_DURATION_SECONDS / functions.BEAT_TOTAL) if functions.BEAT_TOTAL > 0 else 1
TEST_KEY = 60 # Middle C

# --- Test Helper Functions ---
def is_valid_audio_array(audio):
    """Check if the output is a 1D numpy float array."""
    return isinstance(audio, np.ndarray) and audio.ndim == 1 and np.issubdtype(audio.dtype, np.floating)

# --- Basic Function Tests ---

def test_midi_to_freq():
    """Test MIDI note to frequency conversion (A4 = 440Hz)."""
    a4_midi = 69
    expected_freq = 440.0
    assert functions.midi_to_freq(a4_midi) == pytest.approx(expected_freq)

def test_get_allowed_notes():
    """Test if allowed notes are generated correctly for a known scale."""
    # C minor scale starting from C4 (MIDI 60)
    c_minor_notes_midi = [60, 62, 63, 65, 67, 68, 70, 72] # Including octave C5
    allowed = functions.get_allowed_notes(key=60, scale_type="minor", low=60, high=72)
    assert set(allowed) == set(c_minor_notes_midi)

def test_generate_measure_structure():
    """Test the structure and total duration of a generated measure."""
    measure = functions.generate_measure(functions.KEY, functions.ALLOWED_NOTES)
    assert isinstance(measure, list)
    total_duration = 0
    for note in measure:
        assert isinstance(note, tuple)
        assert len(note) == 2
        assert isinstance(note[0], int) # Pitch (MIDI)
        assert isinstance(note[1], float) # Duration
        assert note[0] in functions.ALLOWED_NOTES # Pitch is allowed
        total_duration += note[1]
    # Check if total duration is approximately BEAT_TOTAL
    assert total_duration == pytest.approx(functions.BEAT_TOTAL, abs=0.01)

def test_create_individual_structure():
    """Test the structure of a created individual (list of measures)."""
    individual = functions.create_individual(num_measures=TEST_NUM_MEASURES)
    assert isinstance(individual, list)
    assert len(individual) == TEST_NUM_MEASURES
    for measure in individual:
        assert isinstance(measure, list) # Each element is a measure

# --- Synthesis Tests ---

def test_synthesize_note():
    """Test synthesizing a single note."""
    freq = 440.0
    duration = 0.5
    audio = functions.synthesize_note(freq, duration, sample_rate=TEST_SAMPLE_RATE)
    assert is_valid_audio_array(audio)
    expected_samples = int(duration * TEST_SAMPLE_RATE)
    assert len(audio) == expected_samples

def test_synthesize_melody():
    """Test synthesizing a short melody."""
    melody = [(60, 0.5), (62, 0.5), (63, 1.0)] # C4, D4, Eb4
    audio = functions.synthesize_melody(melody, sample_rate=TEST_SAMPLE_RATE)
    assert is_valid_audio_array(audio)
    expected_duration = sum(d for _, d in melody)
    expected_samples = int(expected_duration * TEST_SAMPLE_RATE)
    # Allow for slight floating point inaccuracies in length calculation
    assert abs(len(audio) - expected_samples) <= 1

# --- Drum Pattern Tests ---

# Test one specific drum pattern function to ensure it runs and returns audio
def test_drum_pattern_1_runs():
    melody = functions.flatten_individual(functions.create_individual(num_measures=TEST_NUM_MEASURES))
    total_duration = sum(d for _, d in melody)
    beat_duration = functions.BASE_TEMPO
    drum_audio = functions.drum_pattern_1(total_duration, beat_duration, melody, sample_rate=TEST_SAMPLE_RATE)
    assert is_valid_audio_array(drum_audio)
    # Check length roughly matches melody duration
    expected_samples = int(total_duration * TEST_SAMPLE_RATE)
    assert abs(len(drum_audio) - expected_samples) <= 1

# --- Effects Tests ---

def test_add_lofi_artifacts():
    """Test that adding artifacts modifies the audio."""
    duration = 1.0
    sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(TEST_SAMPLE_RATE * duration)))
    original_max = np.max(np.abs(sample_audio))

    artifact_audio = functions.add_lofi_artifacts(sample_audio.copy(), TEST_SAMPLE_RATE, artifact_type="crackle", intensity=0.1)

    assert is_valid_audio_array(artifact_audio)
    assert len(artifact_audio) == len(sample_audio)
    # Check that the audio actually changed (RMS difference or max abs diff)
    assert np.max(np.abs(sample_audio - artifact_audio)) > 1e-6 # Should be different
    # Check normalization (max abs value should be close to 1)
    assert np.max(np.abs(artifact_audio)) == pytest.approx(1.0, abs=0.01)


def test_apply_tape_saturation():
    """Test tape saturation application."""
    duration = 1.0
    sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(TEST_SAMPLE_RATE * duration))) * 0.8 # Below 1.0

    saturated_audio = functions.apply_tape_saturation(sample_audio.copy(), saturation_amount=2.0)

    assert is_valid_audio_array(saturated_audio)
    assert len(saturated_audio) == len(sample_audio)
    # Saturation (tanh) should limit max value
    assert np.max(np.abs(saturated_audio)) <= 1.0
    # It should have changed the audio
    assert np.max(np.abs(sample_audio - saturated_audio)) > 1e-6

# --- Main Generation Function Test ---

def test_generate_music_runs_and_returns_correct_types():
    """Test the main generation function to ensure it runs and returns expected types."""
    # Use minimal settings for speed
    audio, sr = functions.generate_music(
        num_measures=TEST_NUM_MEASURES,
        tape_compression=False,
        apply_artifact=False,
        drum_choice="Drum Pattern 1", # Use a specific pattern
        randomize_transpose=False,
        transpose_value=0,
        randomness=False # Reduce randomness for potentially more predictable length
    )

    assert is_valid_audio_array(audio)
    assert isinstance(sr, int)
    assert sr == TEST_SAMPLE_RATE
    # Check length - can be variable due to synthesis details, but should be > 0
    assert len(audio) > 0
    # Optional: Check rough length based on measures and tempo
    # expected_min_duration = TEST_NUM_MEASURES * functions.BEAT_TOTAL * functions.BASE_TEMPO
    # assert len(audio) / sr > expected_min_duration * 0.5 # Allow significant variation

# --- Output Utility Test ---

def test_wav_bytes():
    """Test the WAV byte conversion."""
    sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, TEST_SAMPLE_RATE))
    wav_data = functions.wav_bytes(sample_audio, TEST_SAMPLE_RATE)

    assert isinstance(wav_data, bytes)
    # Check for RIFF header
    assert wav_data.startswith(b'RIFF')

    # Try to read header info using wave module to validate format
    with io.BytesIO(wav_data) as buffer:
        with wave.open(buffer, 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2 # 16-bit
            assert wf.getframerate() == TEST_SAMPLE_RATE
            assert wf.getnframes() == len(sample_audio)