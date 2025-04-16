import random
import numpy as np
import sounddevice as sd
import io
import wave


POPULATION_SIZE = 100
NUM_GENERATIONS = 50
NUM_MEASURES = 4
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
SAMPLE_RATE = 44100
BASE_TEMPO = 0.5
BASE_DURATIONS = [0.25, 0.5, 1.0]
CREATIVE_DURATIONS = [0.75, 1.5]
CREATIVE_PROB = 0.1
BEAT_TOTAL = 4
KEY = 60
SCALE_TYPE = "minor"
SCALE_INTERVALS = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}

def get_allowed_notes(low=48, high=72, key=KEY, scale_type=SCALE_TYPE):
    intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS["major"])
    allowed = []
    for note in range(low, high + 1):
        if ((note - key) % 12) in intervals:
            allowed.append(note)
    return allowed

ALLOWED_NOTES = get_allowed_notes()

def generate_measure(key, allowed_notes, beat_total=BEAT_TOTAL, creative_prob=CREATIVE_PROB):
    notes = []
    note_count = random.randint(3, 6)
    remaining = beat_total
    for i in range(note_count - 1):
        max_possible = remaining - (0.25 * (note_count - i - 1))
        possible = [d for d in BASE_DURATIONS if d <= max_possible]
        if random.random() < creative_prob:
            creative = [d for d in CREATIVE_DURATIONS if d <= max_possible]
            possible.extend(creative)
        duration = random.choice(possible) if possible else 0.25
        remaining -= duration
        pitch = random.choice(allowed_notes)
        notes.append((pitch, duration))
    pitch = random.choice(allowed_notes)
    notes.append((pitch, round(remaining, 3)))
    return notes

def create_individual(num_measures=NUM_MEASURES):
    return [generate_measure(KEY, ALLOWED_NOTES) for _ in range(num_measures)]

def flatten_individual(individual):
    return [note for measure in individual for note in measure]

def fitness(individual):
    melody = flatten_individual(individual)
    score = 0
    intervals = []
    for i in range(1, len(melody)):
        prev_note = melody[i - 1][0]
        curr_note = melody[i][0]
        interval = abs(curr_note - prev_note)
        intervals.append(interval)
        score += 1 if interval <= 5 else - (interval - 5)
    for i in range(1, len(intervals)):
        if abs(intervals[i] - intervals[i - 1]) > 3:
            score -= 1
    if melody[0][0] % 12 == KEY % 12:
        score += 2
    if melody[-1][0] % 12 == KEY % 12:
        score += 2
    return score

def tournament_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=fitness, reverse=True)
    return tournament[0]

def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return [m.copy() for m in parent1], [m.copy() for m in parent2]
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            if random.random() < 0.5:
                individual[i] = generate_measure(KEY, ALLOWED_NOTES)
            else:
                measure = individual[i].copy()
                idx = random.randint(0, len(measure)-1)
                pitch = random.choice(ALLOWED_NOTES)
                duration_options = [d for d in BASE_DURATIONS if d <= BEAT_TOTAL]
                if random.random() < CREATIVE_PROB:
                    duration_options.extend([d for d in CREATIVE_DURATIONS if d <= BEAT_TOTAL])
                new_duration = random.choice(duration_options)
                measure[idx] = (pitch, new_duration)
                total = sum(d for _, d in measure)
                if abs(total - BEAT_TOTAL) < 0.5:
                    diff = BEAT_TOTAL - total
                    pitch_last, dur_last = measure[-1]
                    measure[-1] = (pitch_last, round(dur_last + diff, 3))
                individual[i] = measure
    return individual

def run_ga(num_measures=NUM_MEASURES):
    population = [create_individual(num_measures) for _ in range(POPULATION_SIZE)]
    best_individual, best_fit, stagnant = None, -float("inf"), 0
    global MUTATION_RATE
    mutation_rate = MUTATION_RATE
    for gen in range(NUM_GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
        current_best = max(population, key=fitness)
        current_fit = fitness(current_best)
        if current_fit > best_fit:
            best_fit = current_fit
            best_individual = current_best
            stagnant = 0
        else:
            stagnant += 1
        if stagnant > 5:
            mutation_rate = min(0.5, mutation_rate + 0.05)
        else:
            mutation_rate = MUTATION_RATE
        MUTATION_RATE = mutation_rate
        print(f"Generation {gen+1}: Best Fitness = {best_fit}")
    return best_individual

def generate_adsr_envelope(duration, sample_rate=SAMPLE_RATE):
    total_samples = int(duration * sample_rate)
    attack = min(0.1, duration/4)
    decay = min(0.1, duration/4)
    release = min(0.1, duration/4)
    sustain = duration - attack - decay - release
    if sustain < 0:
        attack = decay = release = duration/3
        sustain = 0
    a = np.linspace(0, 1, int(attack*sample_rate), endpoint=False)
    d = np.linspace(1, 0.8, int(decay*sample_rate), endpoint=False)
    s = np.full(int(sustain*sample_rate), 0.8)
    r = np.linspace(0.8, 0, int(release*sample_rate), endpoint=True)
    env = np.concatenate([a, d, s, r])
    if len(env) < total_samples:
        env = np.pad(env, (0, total_samples-len(env)), 'constant')
    elif len(env) > total_samples:
        env = env[:total_samples]
    return env

def midi_to_freq(midi_note):
    return 440.0 * 2 ** ((midi_note - 69)/12)

def synthesize_note(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    wave = np.sin(2*np.pi*freq*t)
    return wave * generate_adsr_envelope(duration, sample_rate)

def synthesize_melody(melody, sample_rate=SAMPLE_RATE):
    audio = np.array([], dtype=np.float32)
    for note, duration in melody:
        audio = np.concatenate((audio, synthesize_note(midi_to_freq(note), duration, sample_rate)))
    return audio / np.max(np.abs(audio))

def get_chord_from_scale(degree, key, scale_intervals, scale_type=SCALE_TYPE):
    n = len(scale_intervals)
    root = key + scale_intervals[degree % n]
    idx2 = degree + 2
    third = key + scale_intervals[idx2] if idx2 < n else key + scale_intervals[idx2 - n] + 12
    idx3 = degree + 4
    fifth = key + scale_intervals[idx3] if idx3 < n else key + scale_intervals[idx3 - n] + 12
    if scale_type == "minor" and (degree % n) == 4:
        third += 1
    return [root, third, fifth]

def synthesize_chord(chord_notes, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    chord_wave = np.zeros_like(t)
    env = generate_adsr_envelope(duration, sample_rate)
    for note in chord_notes:
        chord_wave += np.sin(2*np.pi*midi_to_freq(note)*t)
    return (chord_wave/len(chord_notes)) * env

def generate_chord_track(total_duration, chord_duration, key, scale_intervals, progression_degrees, sample_rate=SAMPLE_RATE):
    chord_track = np.array([], dtype=np.float32)
    num_chords = int(np.ceil(total_duration/chord_duration))
    prog_len = len(progression_degrees)
    for i in range(num_chords):
        chord = get_chord_from_scale(progression_degrees[i % prog_len], key, scale_intervals, SCALE_TYPE)
        chord_track = np.concatenate((chord_track, synthesize_chord(chord, chord_duration, sample_rate)))
    chord_track = chord_track[:int(total_duration*sample_rate)]
    return chord_track / np.max(np.abs(chord_track))

def synthesize_kick(duration=0.2, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-5*t)
    return np.sin(2*np.pi*100*t) * env

def synthesize_snare(duration=0.15, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-20*t)
    return np.random.uniform(-1, 1, t.shape) * env

def synthesize_hihat(duration=0.1, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-50*t)
    return np.random.uniform(-1, 1, t.shape) * env

def synthesize_bass(note, duration=0.3, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-4*t)
    return np.sin(2*np.pi*(midi_to_freq(note)/2)*t) * env

def get_melody_onsets(melody):
    onsets, time = [], 0
    for note, duration in melody:
        onsets.append((time, note))
        time += duration
    return onsets

def create_drum_loop(total_duration, beat_duration, measure_config, sample_rate=SAMPLE_RATE, randomness=False):
    num_samples = int(total_duration * sample_rate)
    drum_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    num_measures = int(np.ceil(total_duration/measure_duration))
    for m in range(num_measures):
        measure_start = m * measure_duration
        for offset, instrument, multiplier in measure_config:
            event_time = measure_start + offset * beat_duration
            if randomness and random.random() < 0.05:
                event_time += random.uniform(-0.05*beat_duration, 0.05*beat_duration)
                event_time = max(measure_start, min(event_time, measure_start+measure_duration))
            idx = int(event_time * sample_rate)
            drum_sound = instrument(sample_rate=sample_rate)
            if idx+len(drum_sound) < num_samples:
                drum_track[idx:idx+len(drum_sound)] += multiplier * drum_sound
    return drum_track

def drum_pattern_1(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    onsets = get_melody_onsets(melody)
    measure_config = [(0, synthesize_kick, 1.0), (1, synthesize_snare, 1.0), (2, synthesize_kick, 1.0), (3, synthesize_snare, 1.0)]
    for b in range(8):
        measure_config.append((b/2, synthesize_hihat, 0.5))
    drum_track = create_drum_loop(total_duration, beat_duration, measure_config, sample_rate, randomness)
    num_samples = int(total_duration * sample_rate)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    for m in range(int(np.ceil(total_duration/measure_duration))):
        measure_start = m * measure_duration
        chosen_note = next((note for onset_time, note in onsets if onset_time >= measure_start), KEY)
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        if idx+len(bass) < num_samples:
            bass_track[idx:idx+len(bass)] += bass
    full_drum = 0.6*(drum_track/(np.max(np.abs(drum_track))+1e-6)) + 0.8*(bass_track/(np.max(np.abs(bass_track))+1e-6))
    return full_drum/(np.max(np.abs(full_drum))+1e-6)

def drum_pattern_2(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    onsets = get_melody_onsets(melody)
    measure_config = [(0, synthesize_kick, 1.0), (1, synthesize_kick, 1.0), (3, synthesize_kick, 1.0), (2, synthesize_snare, 1.0)]
    for b in range(4):
        measure_config.append((b, synthesize_hihat, 0.3))
    drum_track = create_drum_loop(total_duration, beat_duration, measure_config, sample_rate, randomness)
    num_samples = int(total_duration * sample_rate)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    for m in range(int(np.ceil(total_duration/measure_duration))):
        measure_start = m * measure_duration
        chosen_note = next((note for onset_time, note in onsets if onset_time >= measure_start), KEY)
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        if idx+len(bass) < num_samples:
            bass_track[idx:idx+len(bass)] += bass
    full_drum = 0.7*(drum_track/(np.max(np.abs(drum_track))+1e-6)) + 0.7*(bass_track/(np.max(np.abs(bass_track))+1e-6))
    return full_drum/(np.max(np.abs(full_drum))+1e-6)

def drum_pattern_3(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    onsets = get_melody_onsets(melody)
    measure_config = [(0, synthesize_kick, 1.0), (2.5, synthesize_kick, 1.0), (1.5, synthesize_snare, 1.0), (3.5, synthesize_snare, 1.0)]
    for b in range(8):
        measure_config.append((b/2, synthesize_hihat, 0.4))
    drum_track = create_drum_loop(total_duration, beat_duration, measure_config, sample_rate, randomness)
    num_samples = int(total_duration * sample_rate)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    for m in range(int(np.ceil(total_duration/measure_duration))):
        measure_start = m * measure_duration
        chosen_note = next((note for onset_time, note in onsets if onset_time >= measure_start), KEY)
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        if idx+len(bass) < num_samples:
            bass_track[idx:idx+len(bass)] += bass
    full_drum = 0.6*(drum_track/(np.max(np.abs(drum_track))+1e-6)) + 0.9*(bass_track/(np.max(np.abs(bass_track))+1e-6))
    return full_drum/(np.max(np.abs(full_drum))+1e-6)

def drum_pattern_4(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    onsets = get_melody_onsets(melody)
    measure_config = [(0, synthesize_kick, 1.0), (2, synthesize_snare, 1.0)]
    for b in range(8):
        measure_config.append((b/2, synthesize_hihat, 0.3))
    drum_track = create_drum_loop(total_duration, beat_duration, measure_config, sample_rate, randomness)
    num_samples = int(total_duration * sample_rate)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    for m in range(int(np.ceil(total_duration/measure_duration))):
        measure_start = m * measure_duration
        chosen_note = next((note for onset_time, note in get_melody_onsets(melody) if onset_time >= measure_start), KEY)
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        if idx+len(bass) < num_samples:
            bass_track[idx:idx+len(bass)] += bass
    full_drum = 0.65*(drum_track/(np.max(np.abs(drum_track))+1e-6)) + 0.75*(bass_track/(np.max(np.abs(bass_track))+1e-6))
    return full_drum/(np.max(np.abs(full_drum))+1e-6)

def drum_pattern_5(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    onsets = get_melody_onsets(melody)
    measure_config = [(b, synthesize_kick, 1.0) for b in range(4)]
    measure_config.append((2, synthesize_snare, 1.0))
    for b in range(4):
        measure_config.append(((b+0.5), synthesize_hihat, 0.4))
    drum_track = create_drum_loop(total_duration, beat_duration, measure_config, sample_rate, randomness)
    num_samples = int(total_duration * sample_rate)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    for m in range(int(np.ceil(total_duration/measure_duration))):
        measure_start = m * measure_duration
        chosen_note = next((note for onset_time, note in get_melody_onsets(melody) if onset_time >= measure_start), KEY)
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        if idx+len(bass) < num_samples:
            bass_track[idx:idx+len(bass)] += bass
    full_drum = 0.7*(drum_track/(np.max(np.abs(drum_track))+1e-6)) + 0.8*(bass_track/(np.max(np.abs(bass_track))+1e-6))
    return full_drum/(np.max(np.abs(full_drum))+1e-6)

def generate_random_drum_track(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    patterns = [drum_pattern_1, drum_pattern_2, drum_pattern_3, drum_pattern_4, drum_pattern_5]
    return random.choice(patterns)(total_duration, beat_duration, melody, sample_rate, randomness)

def mix_tracks(melody_audio, chord_audio, drum_audio, melody_volume=0.7, chord_volume=0.3, drum_volume=0.5):
    min_len = min(len(melody_audio), len(chord_audio), len(drum_audio))
    mix = melody_volume * melody_audio[:min_len] + chord_volume * chord_audio[:min_len] + drum_volume * drum_audio[:min_len]
    return mix / np.max(np.abs(mix))

def generate_vinyl_crackle(duration, sample_rate=SAMPLE_RATE, intensity=0.02):
    total_samples = int(duration * sample_rate)
    crackle = np.zeros(total_samples)
    i = 0
    while i < total_samples:
        if random.random() < 0.00005:
            event_length = random.randint(int(0.005 * sample_rate), int(0.02 * sample_rate))
            t = np.linspace(0, event_length/sample_rate, event_length, False)
            envelope = np.exp(-50*t)
            noise = np.random.randn(event_length)*envelope
            end_idx = min(i+event_length, total_samples)
            crackle[i:end_idx] += noise[:end_idx-i]
            i += event_length
        else:
            i += 1
    return intensity * crackle

def generate_fire_crackle(duration, sample_rate=SAMPLE_RATE, intensity=0.02):
    total_samples = int(duration * sample_rate)
    crackle = np.zeros(total_samples)
    i = 0
    while i < total_samples:
        if random.random() < 0.00005:
            event_length = random.randint(int(0.003 * sample_rate), int(0.015 * sample_rate))
            t = np.linspace(0, event_length/sample_rate, event_length, False)
            envelope = np.exp(-70*t)
            noise = np.random.randn(event_length)*envelope
            end_idx = min(i+event_length, total_samples)
            crackle[i:end_idx] += noise[:end_idx-i]
            i += event_length
        else:
            i += 1
    return intensity * crackle

def generate_lofi_artifact(duration, sample_rate=SAMPLE_RATE, artifact_type="crackle", intensity=0.02):
    if artifact_type == "crackle":
        return generate_vinyl_crackle(duration, sample_rate, intensity)
    elif artifact_type == "fire":
        return generate_fire_crackle(duration, sample_rate, intensity)
    return np.zeros(int(duration * sample_rate))

def add_lofi_artifacts(audio, sample_rate=SAMPLE_RATE, artifact_type="crackle", intensity=0.02):
    duration = len(audio)/sample_rate
    artifact = generate_lofi_artifact(duration, sample_rate, artifact_type, intensity)
    mixed = audio + artifact
    return mixed / np.max(np.abs(mixed)+1e-6)

def apply_tape_saturation(audio, saturation_amount=2.0):
    return np.tanh(saturation_amount * audio)

def low_pass_filter(audio, cutoff=5000, sample_rate=SAMPLE_RATE):
    dt = 1 / sample_rate
    RC = 1 / (2 * np.pi * cutoff)
    alpha = dt / (RC + dt)
    filtered = np.zeros_like(audio)
    filtered[0] = audio[0]
    for i in range(1, len(audio)):
        filtered[i] = alpha * audio[i] + (1 - alpha) * filtered[i-1]
    return filtered

def play_audio(audio, sample_rate=SAMPLE_RATE):
    sd.play(audio, sample_rate)
    sd.wait()

def generate_music(tape_compression=False, tape_intensity=0.5, apply_artifact=False, artifact_intensity=0.02, artifact_type="crackle",
                   drum_choice="random", randomize_transpose=False, transpose_value=0, randomness=True, num_measures=NUM_MEASURES):
    best_individual = run_ga(num_measures)
    melody = flatten_individual(best_individual)
    best_notes = [note for note, _ in melody]
    min_note, max_note = min(best_notes), max(best_notes)
    allowed_up = 72 - max_note
    allowed_down = 48 - min_note
    if randomize_transpose:
        possible_trans = [t for t in [-5,-4,-3,-2,-1,1,2,3,4,5] if allowed_down <= t <= allowed_up]
        transposition = random.choice(possible_trans) if possible_trans else 0
    else:
        transposition = transpose_value
    melody = [(note+transposition, duration) for note, duration in melody]
    transposed_key = KEY + transposition
    melody_audio = synthesize_melody(melody)
    total_duration = len(melody_audio)/SAMPLE_RATE
    scale_intervals = SCALE_INTERVALS.get(SCALE_TYPE, SCALE_INTERVALS["major"])
    chord_audio = generate_chord_track(total_duration, 2.0, transposed_key, scale_intervals, [0,3,4,0])
    if drum_choice == "random":
        drum_audio = generate_random_drum_track(total_duration, BASE_TEMPO, melody, SAMPLE_RATE, randomness)
    else:
        drum_map = {"Drum Pattern 1": drum_pattern_1, "Drum Pattern 2": drum_pattern_2,
                    "Drum Pattern 3": drum_pattern_3, "Drum Pattern 4": drum_pattern_4,
                    "Drum Pattern 5": drum_pattern_5}
        drum_func = drum_map.get(drum_choice, generate_random_drum_track)
        drum_audio = drum_func(total_duration, BASE_TEMPO, melody, SAMPLE_RATE, randomness)
    final_audio = mix_tracks(melody_audio, chord_audio, drum_audio)
    if apply_artifact:
        final_audio = add_lofi_artifacts(final_audio, SAMPLE_RATE, artifact_type=artifact_type, intensity=artifact_intensity)
    if tape_compression:
        final_audio = apply_tape_saturation(final_audio, saturation_amount=tape_intensity)
    final_audio = low_pass_filter(final_audio, cutoff=5000, sample_rate=SAMPLE_RATE)
    return final_audio, SAMPLE_RATE

def wav_bytes(audio, sample_rate=SAMPLE_RATE):
    buffer = io.BytesIO()
    n_samples = len(audio)
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        audio_int16 = np.int16(audio/np.max(np.abs(audio)) * 32767)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()
