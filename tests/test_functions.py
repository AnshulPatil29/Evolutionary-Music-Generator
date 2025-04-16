import pytest
import numpy as np
import functions
import wave
import io

TEST_SAMPLE_RATE = functions.SAMPLE_RATE
TEST_DURATION_SECONDS = 2.0
TEST_NUM_MEASURES = max(1, int(TEST_DURATION_SECONDS / functions.BEAT_TOTAL)) if functions.BEAT_TOTAL > 0 else 1
TEST_KEY = 60

def is_valid_audio_array(audio):
    return isinstance(audio, np.ndarray) and audio.ndim == 1 and np.issubdtype(audio.dtype, np.floating)

def test_midi_to_freq():
    a4_midi = 69
    expected_freq = 440.0
    assert functions.midi_to_freq(a4_midi) == pytest.approx(expected_freq)

def test_get_allowed_notes():
    c_minor_notes_midi = [60, 62, 63, 65, 67, 68, 70, 72]
    allowed = functions.get_allowed_notes(key=60, scale_type="minor", low=60, high=72)
    assert set(allowed) == set(c_minor_notes_midi)

def test_generate_measure_structure():
    measure = functions.generate_measure(functions.KEY, functions.ALLOWED_NOTES)
    assert isinstance(measure, list)
    total_duration = 0
    has_notes = False
    for note in measure:
        has_notes = True
        assert isinstance(note, tuple)
        assert len(note) == 2
        assert isinstance(note[0], int)
        assert isinstance(note[1], float)
        assert note[0] in functions.ALLOWED_NOTES
        assert note[1] > 0
        total_duration += note[1]
    assert has_notes
    assert total_duration == pytest.approx(functions.BEAT_TOTAL, abs=0.01)

def test_create_individual_structure():
    individual = functions.create_individual(num_measures=TEST_NUM_MEASURES)
    assert isinstance(individual, list)
    assert len(individual) == TEST_NUM_MEASURES
    for measure in individual:
        assert isinstance(measure, list)
        assert len(measure) > 0

def test_fitness_on_valid_individual():
    individual = functions.create_individual(num_measures=TEST_NUM_MEASURES)
    fitness_score = functions.fitness(individual)
    assert isinstance(fitness_score, (int, float))

def test_fitness_on_empty_individual():
    fitness_score = functions.fitness([])
    assert fitness_score == -float('inf')

def test_synthesize_note():
    freq = 440.0
    duration = 0.5
    audio = functions.synthesize_note(freq, duration, sample_rate=TEST_SAMPLE_RATE)
    assert is_valid_audio_array(audio)
    expected_samples = int(duration * TEST_SAMPLE_RATE)
    assert len(audio) == expected_samples

def test_synthesize_melody():
    melody = [(60, 0.5), (62, 0.5), (63, 1.0)]
    audio = functions.synthesize_melody(melody, sample_rate=TEST_SAMPLE_RATE)
    assert is_valid_audio_array(audio)
    expected_duration = sum(d for _, d in melody)
    expected_samples = int(expected_duration * TEST_SAMPLE_RATE)
    assert abs(len(audio) - expected_samples) <= 1

def test_drum_pattern_1_runs():
    individual = functions.create_individual(num_measures=TEST_NUM_MEASURES)
    melody = functions.flatten_individual(individual)
    if not melody:
        melody = [(TEST_KEY, functions.BEAT_TOTAL)] * TEST_NUM_MEASURES
    total_duration = sum(d for _, d in melody)
    if total_duration <= 0:
        pytest.skip("Skipping drum test because generated melody duration is zero.")
    beat_duration = functions.BASE_TEMPO
    drum_audio = functions.drum_pattern_1(total_duration, beat_duration, melody, sample_rate=TEST_SAMPLE_RATE)
    assert is_valid_audio_array(drum_audio)
    expected_samples = int(total_duration * TEST_SAMPLE_RATE)
    assert abs(len(drum_audio) - expected_samples) <= TEST_SAMPLE_RATE * 0.1

def test_add_lofi_artifacts():
    duration = 1.0
    num_samples = int(TEST_SAMPLE_RATE * duration)
    sample_audio = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples)) +
                    np.sin(2 * np.pi * 880 * np.linspace(0, duration, num_samples)) * 0.5)
    sample_audio = functions._normalize_track(sample_audio)
    artifact_audio = functions.add_lofi_artifacts(sample_audio.copy(), TEST_SAMPLE_RATE, artifact_type="crackle", intensity=0.5)
    assert is_valid_audio_array(artifact_audio)
    assert len(artifact_audio) == len(sample_audio)
    difference = np.sqrt(np.mean((sample_audio - artifact_audio)**2))
    print(f"RMS difference: {difference}")
    assert difference > 1e-3
    assert np.max(np.abs(artifact_audio)) == pytest.approx(1.0, abs=0.05)

def test_apply_tape_saturation():
    duration = 1.0
    sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(TEST_SAMPLE_RATE * duration))) * 0.8
    saturated_audio = functions.apply_tape_saturation(sample_audio.copy(), saturation_amount=2.0)
    assert is_valid_audio_array(saturated_audio)
    assert len(saturated_audio) == len(sample_audio)
    assert np.max(np.abs(saturated_audio)) <= 1.0
    assert np.max(np.abs(sample_audio - saturated_audio)) > 1e-6

def test_generate_music_runs_and_returns_correct_types():
    test_measures = max(1, TEST_NUM_MEASURES)
    audio, sr = functions.generate_music(
        num_measures=test_measures,
        tape_compression=False,
        apply_artifact=False,
        drum_choice="Drum Pattern 1",
        randomize_transpose=False,
        transpose_value=0,
        randomness=False
    )
    assert is_valid_audio_array(audio)
    assert isinstance(sr, int)
    assert sr == TEST_SAMPLE_RATE
    assert len(audio) > 0
    expected_min_duration = test_measures * functions.BEAT_TOTAL * functions.BASE_TEMPO
    assert (len(audio) / sr) >= (expected_min_duration * 0.5)

def test_wav_bytes():
    sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, TEST_SAMPLE_RATE))
    sample_audio = functions._normalize_track(sample_audio)
    wav_data = functions.wav_bytes(sample_audio, TEST_SAMPLE_RATE)
    assert isinstance(wav_data, bytes)
    assert len(wav_data) > 44
    assert wav_data.startswith(b'RIFF')
    with io.BytesIO(wav_data) as buffer:
        with wave.open(buffer, 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == TEST_SAMPLE_RATE
            assert wf.getnframes() == len(sample_audio)

def test_wav_bytes_empty_input():
    empty_audio = np.array([], dtype=np.float32)
    wav_data = functions.wav_bytes(empty_audio, TEST_SAMPLE_RATE)
    assert wav_data == b''