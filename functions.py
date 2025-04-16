# --- START OF FILE functions.py --- (Updated)

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

# --- UPDATED fitness function ---
def fitness(individual):
    melody = flatten_individual(individual)
    # FIX: Handle empty melody case
    if not melody:
        return -float('inf') # Return a very low score if melody is empty

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
    # These lines are now safe due to the check at the start
    if melody[0][0] % 12 == KEY % 12:
        score += 2
    if melody[-1][0] % 12 == KEY % 12:
        score += 2
    return score
# --- END of fitness update ---

def tournament_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=fitness, reverse=True)
    return tournament[0]

# --- UPDATED crossover function ---
def crossover(parent1, parent2):
    # FIX: Handle parents with fewer than 2 measures (crossover not possible/meaningful)
    if len(parent1) < 2 or len(parent2) < 2:
        return [m.copy() for m in parent1], [m.copy() for m in parent2] # Return copies

    # Check crossover rate
    if random.random() > CROSSOVER_RATE:
        return [m.copy() for m in parent1], [m.copy() for m in parent2] # No crossover

    # Choose crossover point (now guaranteed len(parent1) >= 2)
    point = random.randint(1, len(parent1) - 1)

    # Perform crossover
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2
# --- END of crossover update ---

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
    global MUTATION_RATE # Note: Modifying global MUTATION_RATE might have side effects if run multiple times
    mutation_rate = MUTATION_RATE
    for gen in range(NUM_GENERATIONS):
        new_population = []
        # Ensure population size is even for pairing parents
        pop_size_for_pairing = POPULATION_SIZE if POPULATION_SIZE % 2 == 0 else POPULATION_SIZE - 1
        for _ in range(pop_size_for_pairing // 2):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        # Handle odd population size if necessary (e.g., carry over one individual)
        if POPULATION_SIZE % 2 != 0:
             new_population.append(tournament_selection(population)) # Or another strategy

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
            mutation_rate = MUTATION_RATE # Reset if fitness improves
        MUTATION_RATE = mutation_rate # Update global mutation rate
        # print(f"Generation {gen+1}: Best Fitness = {best_fit}")
    # Ensure a valid individual is returned, even if GA didn't find a great one
    if best_individual is None:
        best_individual = create_individual(num_measures) # Create a default one
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
        # Ensure duration is positive before synthesizing
        if duration > 0:
            audio = np.concatenate((audio, synthesize_note(midi_to_freq(note), duration, sample_rate)))
    # Avoid division by zero if audio is silent
    max_abs_audio = np.max(np.abs(audio))
    if max_abs_audio > 1e-6:
        return audio / max_abs_audio
    return audio # Return silent audio as is

def get_chord_from_scale(degree, key, scale_intervals, scale_type=SCALE_TYPE):
    n = len(scale_intervals)
    root_idx = degree % n
    root = key + scale_intervals[root_idx]

    third_idx_rel = root_idx + 2
    third_idx_abs = third_idx_rel % n
    third_octave_offset = third_idx_rel // n
    third = key + scale_intervals[third_idx_abs] + 12 * third_octave_offset

    fifth_idx_rel = root_idx + 4
    fifth_idx_abs = fifth_idx_rel % n
    fifth_octave_offset = fifth_idx_rel // n
    fifth = key + scale_intervals[fifth_idx_abs] + 12 * fifth_octave_offset

    # Minor scale adjustment for the dominant chord (degree 4 -> index 4, which is 7 semitones for root)
    # This logic might need review depending on desired harmony (natural vs harmonic minor etc.)
    # Typically the V chord in minor is major (raised 7th). This happens if degree=4 (0-indexed).
    if scale_type == "minor" and root_idx == 4: # Building chord on 5th degree of minor scale
        third += 1 # Raise the third to make it major

    return [root, third, fifth]


def synthesize_chord(chord_notes, duration, sample_rate=SAMPLE_RATE):
    if duration <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    chord_wave = np.zeros_like(t)
    env = generate_adsr_envelope(duration, sample_rate)
    for note in chord_notes:
        chord_wave += np.sin(2*np.pi*midi_to_freq(note)*t)
    # Avoid division by zero if no notes
    if len(chord_notes) > 0:
        return (chord_wave/len(chord_notes)) * env
    return chord_wave # Return silent wave


def generate_chord_track(total_duration, chord_duration, key, scale_intervals, progression_degrees, sample_rate=SAMPLE_RATE):
    chord_track = np.array([], dtype=np.float32)
    if chord_duration <= 0: # Prevent infinite loop
        return chord_track
    num_chords = int(np.ceil(total_duration/chord_duration))
    prog_len = len(progression_degrees)
    for i in range(num_chords):
        chord = get_chord_from_scale(progression_degrees[i % prog_len], key, scale_intervals, SCALE_TYPE)
        synth_chord_audio = synthesize_chord(chord, chord_duration, sample_rate)
        chord_track = np.concatenate((chord_track, synth_chord_audio))
    # Trim to exact total duration
    final_len = int(total_duration*sample_rate)
    if len(chord_track) > final_len:
        chord_track = chord_track[:final_len]
    elif len(chord_track) < final_len:
         chord_track = np.pad(chord_track, (0, final_len - len(chord_track)), 'constant')

    # Normalize
    max_abs_chord = np.max(np.abs(chord_track))
    if max_abs_chord > 1e-6:
        return chord_track / max_abs_chord
    return chord_track

def synthesize_kick(duration=0.2, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-5*t)
    # Frequency sweep for kick-like sound
    freq = np.linspace(150, 50, len(t))
    return np.sin(2*np.pi*freq*t) * env

def synthesize_snare(duration=0.15, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-20*t)
    # Mix sine wave tone with noise
    tone = np.sin(2*np.pi*180*t) * 0.5
    noise = np.random.uniform(-1, 1, t.shape) * 0.5
    return (tone + noise) * env

def synthesize_hihat(duration=0.1, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-50*t)
    # Filtered noise for hi-hat
    noise = np.random.uniform(-1, 1, t.shape)
    # Simple high-pass effect (difference)
    filtered_noise = np.diff(noise, prepend=0)
    return filtered_noise * env

def synthesize_bass(note, duration=0.3, sample_rate=SAMPLE_RATE):
    if duration <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    env = np.exp(-4*t)
    # Use sub-octave for bass
    freq = midi_to_freq(note) / 2
    # Add a bit of square-like shape for warmth (tanh)
    wave = np.tanh(np.sin(2*np.pi*freq*t) * 1.5)
    return wave * env

def get_melody_onsets(melody):
    onsets, time = [], 0
    for note, duration in melody:
        if duration > 0: # Only consider notes with duration
            onsets.append((time, note))
            time += duration
    return onsets

def create_drum_loop(total_duration, beat_duration, measure_config, sample_rate=SAMPLE_RATE, randomness=False):
    if total_duration <= 0 or beat_duration <= 0:
         return np.array([], dtype=np.float32)
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
                event_time = max(measure_start, min(event_time, measure_start+measure_duration - 0.01)) # Ensure within bounds

            idx = int(event_time * sample_rate)
            drum_sound = instrument(sample_rate=sample_rate)
            end_idx = idx + len(drum_sound)

            # Ensure the sound fits within the track boundaries
            if idx < num_samples and end_idx <= num_samples:
                 drum_track[idx:end_idx] += multiplier * drum_sound
            elif idx < num_samples: # If it goes over, truncate
                 fit_len = num_samples - idx
                 drum_track[idx:] += multiplier * drum_sound[:fit_len]

    return drum_track


def _normalize_track(track):
    """Helper to normalize a track safely."""
    max_abs = np.max(np.abs(track))
    if max_abs > 1e-6:
        return track / max_abs
    return track

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
        # Find the first melody note onset at or after the measure start
        relevant_onsets = [note for onset_time, note in onsets if onset_time >= measure_start]
        chosen_note = relevant_onsets[0] if relevant_onsets else KEY # Default to key if no note found
        bass = synthesize_bass(chosen_note, duration=beat_duration, sample_rate=sample_rate) # Bass note duration tied to beat
        idx = int(measure_start * sample_rate)
        end_idx = idx + len(bass)
        if idx < num_samples and end_idx <= num_samples:
            bass_track[idx:end_idx] += bass
        elif idx < num_samples:
             fit_len = num_samples - idx
             bass_track[idx:] += bass[:fit_len]

    # Normalize individually before mixing
    drum_track_norm = _normalize_track(drum_track)
    bass_track_norm = _normalize_track(bass_track)
    full_drum = 0.6 * drum_track_norm + 0.8 * bass_track_norm
    return _normalize_track(full_drum)

def drum_pattern_2(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    onsets = get_melody_onsets(melody)
    measure_config = [(0, synthesize_kick, 1.0), (1, synthesize_kick, 1.0), (3, synthesize_kick, 1.0), (2, synthesize_snare, 1.0)]
    for b in range(4):
        measure_config.append((b, synthesize_hihat, 0.3)) # Hi-hat on each beat
    drum_track = create_drum_loop(total_duration, beat_duration, measure_config, sample_rate, randomness)

    num_samples = int(total_duration * sample_rate)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    for m in range(int(np.ceil(total_duration/measure_duration))):
        measure_start = m * measure_duration
        relevant_onsets = [note for onset_time, note in onsets if onset_time >= measure_start]
        chosen_note = relevant_onsets[0] if relevant_onsets else KEY
        bass = synthesize_bass(chosen_note, duration=beat_duration*2, sample_rate=sample_rate) # Longer bass note
        idx = int(measure_start * sample_rate)
        end_idx = idx + len(bass)
        if idx < num_samples and end_idx <= num_samples:
            bass_track[idx:end_idx] += bass
        elif idx < num_samples:
             fit_len = num_samples - idx
             bass_track[idx:] += bass[:fit_len]

    drum_track_norm = _normalize_track(drum_track)
    bass_track_norm = _normalize_track(bass_track)
    full_drum = 0.7 * drum_track_norm + 0.7 * bass_track_norm
    return _normalize_track(full_drum)


def drum_pattern_3(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    onsets = get_melody_onsets(melody)
    measure_config = [(0, synthesize_kick, 1.0), (2.5, synthesize_kick, 0.8), (1.5, synthesize_snare, 1.0), (3.5, synthesize_snare, 0.7)]
    for b in range(8): # 8th note hi-hats
        measure_config.append((b/2, synthesize_hihat, 0.4))
    drum_track = create_drum_loop(total_duration, beat_duration, measure_config, sample_rate, randomness)

    num_samples = int(total_duration * sample_rate)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    for m in range(int(np.ceil(total_duration/measure_duration))):
        measure_start = m * measure_duration
        relevant_onsets = [note for onset_time, note in onsets if onset_time >= measure_start]
        chosen_note = relevant_onsets[0] if relevant_onsets else KEY
        # Vary bass duration
        bass_duration = random.choice([beat_duration, beat_duration*1.5]) if randomness else beat_duration
        bass = synthesize_bass(chosen_note, duration=bass_duration, sample_rate=sample_rate)
        idx = int(measure_start * sample_rate)
        end_idx = idx + len(bass)
        if idx < num_samples and end_idx <= num_samples:
            bass_track[idx:end_idx] += bass
        elif idx < num_samples:
             fit_len = num_samples - idx
             bass_track[idx:] += bass[:fit_len]

    drum_track_norm = _normalize_track(drum_track)
    bass_track_norm = _normalize_track(bass_track)
    full_drum = 0.6 * drum_track_norm + 0.9 * bass_track_norm # Bass heavier mix
    return _normalize_track(full_drum)


def drum_pattern_4(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    onsets = get_melody_onsets(melody)
    # Simpler pattern
    measure_config = [(0, synthesize_kick, 1.0), (2, synthesize_snare, 1.0)]
    for b in range(8): # 8th note hi-hats
        measure_config.append((b/2, synthesize_hihat, 0.3))
    drum_track = create_drum_loop(total_duration, beat_duration, measure_config, sample_rate, randomness)

    num_samples = int(total_duration * sample_rate)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    for m in range(int(np.ceil(total_duration/measure_duration))):
        measure_start = m * measure_duration
        relevant_onsets = [note for onset_time, note in onsets if onset_time >= measure_start]
        chosen_note = relevant_onsets[0] if relevant_onsets else KEY
        bass = synthesize_bass(chosen_note, duration=beat_duration * 4, sample_rate=sample_rate) # Whole note bass
        idx = int(measure_start * sample_rate)
        end_idx = idx + len(bass)
        if idx < num_samples and end_idx <= num_samples:
            bass_track[idx:end_idx] += bass
        elif idx < num_samples:
             fit_len = num_samples - idx
             bass_track[idx:] += bass[:fit_len]

    drum_track_norm = _normalize_track(drum_track)
    bass_track_norm = _normalize_track(bass_track)
    full_drum = 0.65 * drum_track_norm + 0.75 * bass_track_norm
    return _normalize_track(full_drum)


def drum_pattern_5(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    onsets = get_melody_onsets(melody)
    # Four on the floor kick
    measure_config = [(b, synthesize_kick, 1.0) for b in range(4)]
    measure_config.append((2, synthesize_snare, 1.0)) # Snare on 2 (or 3 if count starts at 1)
    # Off-beat hi-hats
    for b in range(4):
        measure_config.append(((b+0.5), synthesize_hihat, 0.4))
    drum_track = create_drum_loop(total_duration, beat_duration, measure_config, sample_rate, randomness)

    num_samples = int(total_duration * sample_rate)
    bass_track = np.zeros(num_samples)
    measure_duration = 4 * beat_duration
    for m in range(int(np.ceil(total_duration/measure_duration))):
        measure_start = m * measure_duration
        relevant_onsets = [note for onset_time, note in onsets if onset_time >= measure_start]
        chosen_note = relevant_onsets[0] if relevant_onsets else KEY
        bass = synthesize_bass(chosen_note, duration=beat_duration, sample_rate=sample_rate)
        # Place bass on beats 1 and 3 (or 0 and 2)
        for beat_offset in [0, 2]:
             idx = int((measure_start + beat_offset * beat_duration) * sample_rate)
             end_idx = idx + len(bass)
             if idx < num_samples and end_idx <= num_samples:
                 bass_track[idx:end_idx] += bass
             elif idx < num_samples:
                 fit_len = num_samples - idx
                 bass_track[idx:] += bass[:fit_len]

    drum_track_norm = _normalize_track(drum_track)
    bass_track_norm = _normalize_track(bass_track)
    full_drum = 0.7 * drum_track_norm + 0.8 * bass_track_norm
    return _normalize_track(full_drum)


def generate_random_drum_track(total_duration, beat_duration, melody, sample_rate=SAMPLE_RATE, randomness=False):
    patterns = [drum_pattern_1, drum_pattern_2, drum_pattern_3, drum_pattern_4, drum_pattern_5]
    chosen_pattern = random.choice(patterns)
    return chosen_pattern(total_duration, beat_duration, melody, sample_rate, randomness)

def mix_tracks(melody_audio, chord_audio, drum_audio, melody_volume=0.7, chord_volume=0.3, drum_volume=0.5):
    # Find the minimum length among all tracks
    min_len = min(len(melody_audio), len(chord_audio), len(drum_audio))

    # Ensure minimum length is positive before slicing
    if min_len <= 0:
        return np.array([], dtype=np.float32) # Return empty if any track is effectively empty

    # Mix the tracks, slicing to the minimum length
    mix = (melody_volume * _normalize_track(melody_audio[:min_len]) +
           chord_volume * _normalize_track(chord_audio[:min_len]) +
           drum_volume * _normalize_track(drum_audio[:min_len]))

    # Normalize the final mix
    return _normalize_track(mix)

# --- Artifact Generation Functions ---

def generate_vinyl_crackle(duration, sample_rate=SAMPLE_RATE, intensity=0.02, probability=0.0005):
    if duration <= 0: return np.zeros(0)
    total_samples = int(duration * sample_rate)
    crackle = np.zeros(total_samples)
    i = 0
    while i < total_samples:
        if random.random() < probability: # Adjusted probability
            event_length = random.randint(int(0.005 * sample_rate), int(0.02 * sample_rate))
            t = np.linspace(0, event_length/sample_rate, event_length, False)
            envelope = np.exp(-random.uniform(30, 70)*t) # Randomize decay
            noise = np.random.randn(event_length)*envelope
            end_idx = min(i+event_length, total_samples)
            crackle[i:end_idx] += noise[:end_idx-i]
            i += event_length # Move index past the event
        else:
            i += random.randint(1, 5) # Move index forward randomly even if no event
    return intensity * crackle

def generate_fire_crackle(duration, sample_rate=SAMPLE_RATE, intensity=0.02, probability=0.0006):
    if duration <= 0: return np.zeros(0)
    total_samples = int(duration * sample_rate)
    crackle = np.zeros(total_samples)
    i = 0
    while i < total_samples:
         if random.random() < probability: # Adjusted probability
            event_length = random.randint(int(0.003 * sample_rate), int(0.015 * sample_rate))
            t = np.linspace(0, event_length/sample_rate, event_length, False)
            envelope = np.exp(-random.uniform(50, 90)*t) # Randomize decay
            noise = np.random.randn(event_length)*envelope
            end_idx = min(i+event_length, total_samples)
            crackle[i:end_idx] += noise[:end_idx-i]
            i += event_length
         else:
             i += random.randint(1, 4)
    return intensity * crackle

def generate_lofi_artifact(duration, sample_rate=SAMPLE_RATE, artifact_type="crackle", intensity=0.02):
    if artifact_type == "crackle":
        # Increase probability for shorter test durations if needed, pass intensity
        prob = 0.005 if duration < 2.0 else 0.0005 # Example adjustment
        return generate_vinyl_crackle(duration, sample_rate, intensity=intensity, probability=prob)
    elif artifact_type == "fire":
        prob = 0.006 if duration < 2.0 else 0.0006 # Example adjustment
        return generate_fire_crackle(duration, sample_rate, intensity=intensity, probability=prob)
    return np.zeros(int(duration * sample_rate))

def add_lofi_artifacts(audio, sample_rate=SAMPLE_RATE, artifact_type="crackle", intensity=0.02):
    duration = len(audio)/sample_rate
    if duration <= 0: return audio # Return original if no duration
    artifact = generate_lofi_artifact(duration, sample_rate, artifact_type, intensity)
    # Ensure artifact is same length as audio
    if len(artifact) != len(audio):
         # Pad or truncate artifact if necessary (shouldn't happen with current logic)
         target_len = len(audio)
         if len(artifact) > target_len:
             artifact = artifact[:target_len]
         else:
             artifact = np.pad(artifact, (0, target_len - len(artifact)), 'constant')

    mixed = audio + artifact
    return _normalize_track(mixed) # Use helper for safe normalization

# --- Effects ---

def apply_tape_saturation(audio, saturation_amount=2.0):
    # Apply soft clipping using tanh
    return np.tanh(saturation_amount * audio)

def low_pass_filter(audio, cutoff=5000, sample_rate=SAMPLE_RATE):
    if len(audio) == 0: return audio # Handle empty audio
    dt = 1 / sample_rate
    RC = 1 / (2 * np.pi * cutoff)
    alpha = dt / (RC + dt)
    filtered = np.zeros_like(audio)
    filtered[0] = alpha * audio[0] # Initialize first sample slightly differently
    for i in range(1, len(audio)):
        filtered[i] = alpha * audio[i] + (1 - alpha) * filtered[i-1]
    return filtered

# --- Playback (Requires sounddevice) ---

def play_audio(audio, sample_rate=SAMPLE_RATE):
    """Plays audio if sounddevice is available and audio is not empty."""
    if len(audio) > 0:
        try:
            sd.play(audio, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Could not play audio: {e}")
            print("Ensure you have a working audio output device and 'sounddevice' installed.")
    else:
        print("Audio is empty, nothing to play.")

# --- Main Generation Function ---

def generate_music(tape_compression=False, tape_intensity=0.5, apply_artifact=False, artifact_intensity=0.02, artifact_type="crackle",
                   drum_choice="random", randomize_transpose=False, transpose_value=0, randomness=True, num_measures=NUM_MEASURES):

    if num_measures <= 0:
         print("Warning: num_measures is zero or negative. No music generated.")
         return np.array([], dtype=np.float32), SAMPLE_RATE

    best_individual = run_ga(num_measures)

    # Ensure GA returned a valid individual
    if not best_individual:
         print("Warning: Genetic algorithm failed to produce an individual.")
         best_individual = create_individual(num_measures) # Fallback

    melody = flatten_individual(best_individual)

    if not melody:
         print("Warning: Generated individual resulted in an empty melody.")
         return np.array([], dtype=np.float32), SAMPLE_RATE

    # Transposition logic
    best_notes = [note for note, _ in melody]
    min_note, max_note = min(best_notes), max(best_notes)
    allowed_up = 72 - max_note   # Max allowed note is MIDI 72 (C5)
    allowed_down = 48 - min_note # Min allowed note is MIDI 48 (C3)

    if randomize_transpose:
        # Define possible transpose steps, filter by allowed range
        possible_trans = [t for t in range(-5, 6) if t != 0 and allowed_down <= t <= allowed_up]
        transposition = random.choice(possible_trans) if possible_trans else 0
    else:
        # Ensure fixed transpose value is within allowed range
        transposition = max(allowed_down, min(transpose_value, allowed_up))

    # Apply transposition
    melody = [(note + transposition, duration) for note, duration in melody]
    transposed_key = KEY + transposition

    # Synthesize Melody
    melody_audio = synthesize_melody(melody, SAMPLE_RATE)
    total_duration = len(melody_audio) / SAMPLE_RATE

    if total_duration <= 0:
         print("Warning: Synthesized melody has zero duration.")
         return np.array([], dtype=np.float32), SAMPLE_RATE

    # Synthesize Chords
    scale_intervals = SCALE_INTERVALS.get(SCALE_TYPE, SCALE_INTERVALS["major"])
    # Common Lo-fi progression (e.g., I-vi-IV-V in Major, adapted for minor -> i-VI-iv-V)
    # Using degrees relative to the scale: 0=i, 5=VI, 3=iv, 4=V (often major V in minor key)
    progression_degrees = [0, 5, 3, 4]
    chord_audio = generate_chord_track(total_duration, 2.0, transposed_key, scale_intervals, progression_degrees, SAMPLE_RATE) # 2 sec per chord

    # Synthesize Drums
    if drum_choice == "random":
        drum_audio = generate_random_drum_track(total_duration, BASE_TEMPO, melody, SAMPLE_RATE, randomness)
    else:
        drum_map = {"Drum Pattern 1": drum_pattern_1, "Drum Pattern 2": drum_pattern_2,
                    "Drum Pattern 3": drum_pattern_3, "Drum Pattern 4": drum_pattern_4,
                    "Drum Pattern 5": drum_pattern_5}
        drum_func = drum_map.get(drum_choice, generate_random_drum_track) # Fallback to random
        drum_audio = drum_func(total_duration, BASE_TEMPO, melody, SAMPLE_RATE, randomness)

    # Mix Tracks
    final_audio = mix_tracks(melody_audio, chord_audio, drum_audio)

    # Apply Effects
    if apply_artifact:
        final_audio = add_lofi_artifacts(final_audio, SAMPLE_RATE, artifact_type=artifact_type, intensity=artifact_intensity)
    if tape_compression:
        # Apply saturation before filtering for a more 'driven' sound
        final_audio = apply_tape_saturation(final_audio, saturation_amount=tape_intensity)

    # Apply final low-pass filter for Lo-fi feel
    final_audio = low_pass_filter(final_audio, cutoff=random.uniform(4000, 6000) if randomness else 5000, sample_rate=SAMPLE_RATE)

    return _normalize_track(final_audio), SAMPLE_RATE # Ensure final output is normalized


def wav_bytes(audio, sample_rate=SAMPLE_RATE):
    """Converts numpy audio array to WAV byte stream."""
    buffer = io.BytesIO()
    # Ensure audio is not empty
    if len(audio) == 0:
        # Write a minimal valid WAV header for silence if needed, or return empty bytes
        # For simplicity, let's return empty bytes, calling code should handle it.
         return b''

    n_samples = len(audio)
    # Normalize audio before converting to int16 to prevent clipping
    normalized_audio = _normalize_track(audio)
    audio_int16 = np.int16(normalized_audio * 32767)

    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)       # Mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return buffer.getvalue()

# --- END OF FILE functions.py ---