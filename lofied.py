import random
import numpy as np
import sounddevice as sd

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
    flat = []
    for measure in individual:
        flat.extend(measure)
    return flat

def fitness(individual):
    melody = flatten_individual(individual)
    score = 0
    intervals = []
    for i in range(1, len(melody)):
        prev_note = melody[i - 1][0]
        curr_note = melody[i][0]
        interval = abs(curr_note - prev_note)
        intervals.append(interval)
        if interval <= 5:
            score += 1
        else:
            score -= (interval - 5)
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
    point = random.randint(1, NUM_MEASURES - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

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

def run_ga():
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_individual = None
    best_fit = -float("inf")
    stagnant = 0
    global MUTATION_RATE
    mutation_rate = MUTATION_RATE
    for gen in range(NUM_GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
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
    env = generate_adsr_envelope(duration, sample_rate)
    return wave * env

def synthesize_melody(melody, sample_rate=SAMPLE_RATE):
    audio = np.array([], dtype=np.float32)
    for note, duration in melody:
        freq = midi_to_freq(note)
        wave = synthesize_note(freq, duration, sample_rate)
        audio = np.concatenate((audio, wave))
    audio = audio / np.max(np.abs(audio))
    return audio

def get_chord_from_scale(degree, key, scale_intervals, scale_type=SCALE_TYPE):
    n = len(scale_intervals)
    root = key + scale_intervals[degree % n]
    idx2 = degree + 2
    if idx2 < n:
        third = key + scale_intervals[idx2]
    else:
        third = key + scale_intervals[idx2 - n] + 12
    idx3 = degree + 4
    if idx3 < n:
        fifth = key + scale_intervals[idx3]
    else:
        fifth = key + scale_intervals[idx3 - n] + 12
    if scale_type == "minor" and (degree % n) == 4:
        third += 1
    return [root, third, fifth]

def synthesize_chord(chord_notes, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    chord_wave = np.zeros_like(t)
    env = generate_adsr_envelope(duration, sample_rate)
    for note in chord_notes:
        freq = midi_to_freq(note)
        chord_wave += np.sin(2*np.pi*freq*t)
    chord_wave = chord_wave / len(chord_notes)
    return chord_wave * env

def generate_chord_track(total_duration, chord_duration, key, scale_intervals, progression_degrees, sample_rate=SAMPLE_RATE):
    chord_track = np.array([], dtype=np.float32)
    num_chords = int(np.ceil(total_duration/chord_duration))
    prog_len = len(progression_degrees)
    for i in range(num_chords):
        degree = progression_degrees[i % prog_len]
        chord = get_chord_from_scale(degree, key, scale_intervals, SCALE_TYPE)
        chord_wave = synthesize_chord(chord, chord_duration, sample_rate)
        chord_track = np.concatenate((chord_track, chord_wave))
    chord_track = chord_track[:int(total_duration*sample_rate)]
    chord_track = chord_track / np.max(np.abs(chord_track))
    return chord_track

def synthesize_kick(duration=0.2, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    freq = 100
    env = np.exp(-5*t)
    return np.sin(2*np.pi*freq*t) * env

def synthesize_snare(duration=0.15, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-20*t)
    noise = np.random.uniform(-1, 1, t.shape)
    return noise * env

def synthesize_hihat(duration=0.1, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-50*t)
    noise = np.random.uniform(-1, 1, t.shape)
    return noise * env

def synthesize_bass(note, duration=0.3, sample_rate=SAMPLE_RATE):
    freq = midi_to_freq(note) / 2
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-4*t)
    return np.sin(2*np.pi*freq*t) * env

def get_melody_onsets(melody):
    onsets = []
    time = 0
    for note, duration in melody:
        onsets.append((time, note))
        time += duration
    return onsets

def drum_pattern_1(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE):
    num_samples = int(total_duration * sample_rate)
    drum_track = np.zeros(num_samples)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    onsets = get_melody_onsets(melody)
    num_measures = int(np.ceil(total_duration / measure_duration))
    for m in range(num_measures):
        measure_start = m * measure_duration
        for b in range(4):
            beat_time = measure_start + b * beat_duration
            idx = int(beat_time * sample_rate)
            if b in [0, 2]:
                kick = synthesize_kick(sample_rate=sample_rate)
                end_idx = idx + len(kick)
                if end_idx < num_samples:
                    drum_track[idx:end_idx] += kick
            if b in [1, 3]:
                snare = synthesize_snare(sample_rate=sample_rate)
                end_idx = idx + len(snare)
                if end_idx < num_samples:
                    drum_track[idx:end_idx] += snare
        for b in range(8):
            beat_time = measure_start + b * (beat_duration / 2)
            idx = int(beat_time * sample_rate)
            hihat = synthesize_hihat(sample_rate=sample_rate)
            end_idx = idx + len(hihat)
            if end_idx < num_samples:
                drum_track[idx:end_idx] += hihat * 0.5
        chosen_note = None
        for onset_time, note in onsets:
            if onset_time >= measure_start:
                chosen_note = note
                break
        if chosen_note is None:
            chosen_note = KEY
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        end_idx = idx + len(bass)
        if end_idx < num_samples:
            bass_track[idx:end_idx] += bass
    drum_track = drum_track / (np.max(np.abs(drum_track)) + 1e-6)
    bass_track = bass_track / (np.max(np.abs(bass_track)) + 1e-6)
    full_drum = 0.6 * drum_track + 0.8 * bass_track
    full_drum = full_drum / (np.max(np.abs(full_drum)) + 1e-6)
    return full_drum

def drum_pattern_2(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE):
    num_samples = int(total_duration * sample_rate)
    drum_track = np.zeros(num_samples)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    onsets = get_melody_onsets(melody)
    num_measures = int(np.ceil(total_duration / measure_duration))
    for m in range(num_measures):
        measure_start = m * measure_duration
        for b in [0, 1]:
            beat_time = measure_start + b * beat_duration
            idx = int(beat_time * sample_rate)
            kick = synthesize_kick(sample_rate=sample_rate)
            end_idx = idx + len(kick)
            if end_idx < num_samples:
                drum_track[idx:end_idx] += kick
        beat_time = measure_start + 3 * beat_duration
        idx = int(beat_time * sample_rate)
        kick = synthesize_kick(sample_rate=sample_rate)
        end_idx = idx + len(kick)
        if end_idx < num_samples:
            drum_track[idx:end_idx] += kick
        beat_time = measure_start + 2 * beat_duration
        idx = int(beat_time * sample_rate)
        snare = synthesize_snare(sample_rate=sample_rate)
        end_idx = idx + len(snare)
        if end_idx < num_samples:
            drum_track[idx:end_idx] += snare
        for b in range(4):
            beat_time = measure_start + b * beat_duration
            idx = int(beat_time * sample_rate)
            hihat = synthesize_hihat(sample_rate=sample_rate)
            end_idx = idx + len(hihat)
            if end_idx < num_samples:
                drum_track[idx:end_idx] += hihat * 0.3
        chosen_note = None
        for onset_time, note in onsets:
            if onset_time >= measure_start:
                chosen_note = note
                break
        if chosen_note is None:
            chosen_note = KEY
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        end_idx = idx + len(bass)
        if end_idx < num_samples:
            bass_track[idx:end_idx] += bass
    drum_track = drum_track / (np.max(np.abs(drum_track)) + 1e-6)
    bass_track = bass_track / (np.max(np.abs(bass_track)) + 1e-6)
    full_drum = 0.7 * drum_track + 0.7 * bass_track
    full_drum = full_drum / (np.max(np.abs(full_drum)) + 1e-6)
    return full_drum

def drum_pattern_3(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE):
    num_samples = int(total_duration * sample_rate)
    drum_track = np.zeros(num_samples)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    onsets = get_melody_onsets(melody)
    num_measures = int(np.ceil(total_duration / measure_duration))
    for m in range(num_measures):
        measure_start = m * measure_duration
        beat_time = measure_start
        idx = int(beat_time * sample_rate)
        kick = synthesize_kick(sample_rate=sample_rate)
        end_idx = idx + len(kick)
        if end_idx < num_samples:
            drum_track[idx:end_idx] += kick
        beat_time = measure_start + 2.5 * beat_duration
        idx = int(beat_time * sample_rate)
        kick = synthesize_kick(sample_rate=sample_rate)
        end_idx = idx + len(kick)
        if end_idx < num_samples:
            drum_track[idx:end_idx] += kick
        for b in [1.5, 3.5]:
            beat_time = measure_start + b * beat_duration
            idx = int(beat_time * sample_rate)
            snare = synthesize_snare(sample_rate=sample_rate)
            end_idx = idx + len(snare)
            if end_idx < num_samples:
                drum_track[idx:end_idx] += snare
        for b in range(8):
            beat_time = measure_start + b * (beat_duration / 2)
            idx = int(beat_time * sample_rate)
            hihat = synthesize_hihat(sample_rate=sample_rate)
            end_idx = idx + len(hihat)
            if end_idx < num_samples:
                drum_track[idx:end_idx] += hihat * 0.4
        chosen_note = None
        for onset_time, note in onsets:
            if onset_time >= measure_start:
                chosen_note = note
                break
        if chosen_note is None:
            chosen_note = KEY
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        end_idx = idx + len(bass)
        if end_idx < num_samples:
            bass_track[idx:end_idx] += bass
    drum_track = drum_track / (np.max(np.abs(drum_track)) + 1e-6)
    bass_track = bass_track / (np.max(np.abs(bass_track)) + 1e-6)
    full_drum = 0.6 * drum_track + 0.9 * bass_track
    full_drum = full_drum / (np.max(np.abs(full_drum)) + 1e-6)
    return full_drum

def drum_pattern_4(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE):
    num_samples = int(total_duration * sample_rate)
    drum_track = np.zeros(num_samples)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    onsets = get_melody_onsets(melody)
    num_measures = int(np.ceil(total_duration / measure_duration))
    for m in range(num_measures):
        measure_start = m * measure_duration
        beat_time = measure_start
        idx = int(beat_time * sample_rate)
        kick = synthesize_kick(sample_rate=sample_rate)
        end_idx = idx + len(kick)
        if end_idx < num_samples:
            drum_track[idx:end_idx] += kick
        beat_time = measure_start + 2 * beat_duration
        idx = int(beat_time * sample_rate)
        snare = synthesize_snare(sample_rate=sample_rate)
        end_idx = idx + len(snare)
        if end_idx < num_samples:
            drum_track[idx:end_idx] += snare
        for b in range(8):
            beat_time = measure_start + b * (beat_duration / 2)
            idx = int(beat_time * sample_rate)
            hihat = synthesize_hihat(sample_rate=sample_rate)
            end_idx = idx + len(hihat)
            if end_idx < num_samples:
                drum_track[idx:end_idx] += hihat * 0.3
        chosen_note = None
        for onset_time, note in onsets:
            if onset_time >= measure_start:
                chosen_note = note
                break
        if chosen_note is None:
            chosen_note = KEY
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        end_idx = idx + len(bass)
        if end_idx < num_samples:
            bass_track[idx:end_idx] += bass
    drum_track = drum_track / (np.max(np.abs(drum_track)) + 1e-6)
    bass_track = bass_track / (np.max(np.abs(bass_track)) + 1e-6)
    full_drum = 0.65 * drum_track + 0.75 * bass_track
    full_drum = full_drum / (np.max(np.abs(full_drum)) + 1e-6)
    return full_drum

def drum_pattern_5(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE):
    num_samples = int(total_duration * sample_rate)
    drum_track = np.zeros(num_samples)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    onsets = get_melody_onsets(melody)
    num_measures = int(np.ceil(total_duration / measure_duration))
    for m in range(num_measures):
        measure_start = m * measure_duration
        for b in range(4):
            beat_time = measure_start + b * beat_duration
            idx = int(beat_time * sample_rate)
            kick = synthesize_kick(sample_rate=sample_rate)
            end_idx = idx + len(kick)
            if end_idx < num_samples:
                drum_track[idx:end_idx] += kick
        beat_time = measure_start + 2 * beat_duration
        idx = int(beat_time * sample_rate)
        snare = synthesize_snare(sample_rate=sample_rate)
        end_idx = idx + len(snare)
        if end_idx < num_samples:
            drum_track[idx:end_idx] += snare
        for b in range(4):
            beat_time = measure_start + b * beat_duration + (beat_duration / 2)
            idx = int(beat_time * sample_rate)
            hihat = synthesize_hihat(sample_rate=sample_rate)
            end_idx = idx + len(hihat)
            if end_idx < num_samples:
                drum_track[idx:end_idx] += hihat * 0.4
        chosen_note = None
        for onset_time, note in onsets:
            if onset_time >= measure_start:
                chosen_note = note
                break
        if chosen_note is None:
            chosen_note = KEY
        bass = synthesize_bass(chosen_note, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        end_idx = idx + len(bass)
        if end_idx < num_samples:
            bass_track[idx:end_idx] += bass
    drum_track = drum_track / (np.max(np.abs(drum_track)) + 1e-6)
    bass_track = bass_track / (np.max(np.abs(bass_track)) + 1e-6)
    full_drum = 0.7 * drum_track + 0.8 * bass_track
    full_drum = full_drum / (np.max(np.abs(full_drum)) + 1e-6)
    return full_drum

def generate_random_drum_track(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE):
    patterns = [drum_pattern_1, drum_pattern_2, drum_pattern_3, drum_pattern_4, drum_pattern_5]
    chosen_pattern = random.choice(patterns)
    return chosen_pattern(total_duration, beat_duration, melody, sample_rate)

def mix_tracks(melody_audio, chord_audio, drum_audio, melody_volume=0.7, chord_volume=0.3, drum_volume=0.5):
    min_len = min(len(melody_audio), len(chord_audio), len(drum_audio))
    melody_audio = melody_audio[:min_len]
    chord_audio = chord_audio[:min_len]
    drum_audio = drum_audio[:min_len]
    mix = melody_volume * melody_audio + chord_volume * chord_audio + drum_volume * drum_audio
    mix = mix / np.max(np.abs(mix))
    return mix

def generate_vinyl_crackle(duration, sample_rate=SAMPLE_RATE, intensity=0.02):
    total_samples = int(duration * sample_rate)
    crackle = np.zeros(total_samples)
    event_probability = 0.00005
    i = 0
    while i < total_samples:
        if random.random() < event_probability:
            event_length = random.randint(int(0.005 * sample_rate), int(0.02 * sample_rate))
            t = np.linspace(0, event_length/sample_rate, event_length, False)
            envelope = np.exp(-t * 50)
            noise = np.random.randn(event_length) * envelope
            end_idx = min(i + event_length, total_samples)
            crackle[i:end_idx] += noise[:end_idx-i]
            i += event_length
        else:
            i += 1
    return intensity * crackle

def generate_fire_crackle(duration, sample_rate=SAMPLE_RATE, intensity=0.02):
    total_samples = int(duration * sample_rate)
    crackle = np.zeros(total_samples)
    event_probability = 0.00005
    i = 0
    while i < total_samples:
        if random.random() < event_probability:
            event_length = random.randint(int(0.003 * sample_rate), int(0.015 * sample_rate))
            t = np.linspace(0, event_length/sample_rate, event_length, False)
            envelope = np.exp(-t * 70)
            noise = np.random.randn(event_length) * envelope
            end_idx = min(i + event_length, total_samples)
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
    else:
        return np.zeros(int(duration * sample_rate))

def add_lofi_artifacts(audio, sample_rate=SAMPLE_RATE, artifact_type="crackle", intensity=0.02):
    duration = len(audio) / sample_rate
    artifact = generate_lofi_artifact(duration, sample_rate, artifact_type, intensity)
    mixed = audio + artifact
    mixed = mixed / np.max(np.abs(mixed) + 1e-6)
    return mixed

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

if __name__ == "__main__":
    best_individual = run_ga()
    melody = flatten_individual(best_individual)
    print("Best Melody (MIDI Note, Duration):")
    for note, duration in melody:
        print(f"Note: {note}, Duration: {duration}")
    best_notes = [note for note, dur in melody]
    min_note = min(best_notes)
    max_note = max(best_notes)
    allowed_up = 72 - max_note
    allowed_down = 48 - min_note
    possible_trans = [t for t in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5] if allowed_down <= t <= allowed_up]
    transposition = random.choice(possible_trans) if possible_trans else 0
    print("Transposition offset:", transposition)
    melody = [(note + transposition, dur) for note, dur in melody]
    transposed_key = KEY + transposition
    melody[-1] = (transposed_key, melody[-1][1])
    melody_audio = synthesize_melody(melody)
    total_duration = len(melody_audio) / SAMPLE_RATE
    scale_intervals = SCALE_INTERVALS.get(SCALE_TYPE, SCALE_INTERVALS["major"])
    chord_duration = 2.0
    chord_audio = generate_chord_track(total_duration, chord_duration, transposed_key, scale_intervals, [0, 3, 4, 0])
    drum_audio = generate_random_drum_track(total_duration, BASE_TEMPO, melody)
    final_audio = mix_tracks(melody_audio, chord_audio, drum_audio)
    final_audio = add_lofi_artifacts(final_audio, artifact_type="crackle", intensity=0.02)
    # final_audio = apply_tape_saturation(final_audio, saturation_amount=2.0)
    final_audio = low_pass_filter(final_audio, cutoff=5000, sample_rate=SAMPLE_RATE)
    print("Playing generated music with enhanced lo-fi effects...")
    play_audio(final_audio, sample_rate=SAMPLE_RATE)
