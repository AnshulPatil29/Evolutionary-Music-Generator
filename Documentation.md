# Evolutionary Lofi Music Generation - Documentation

## Overview

This Python script utilizes a **Genetic Algorithm (GA)** to automatically compose short musical melodies. It then synthesizes these melodies, generates accompanying diatonic chord progressions and rhythmic drum/bass patterns, and applies characteristic **Lofi effects** (low-pass filtering, simulated tape saturation, crackle) to produce a complete musical piece.

## Core Concepts

1.  **Genetic Algorithm (GA):** An optimization technique inspired by natural selection. It evolves a *population* of candidate solutions (melodies, called *individuals*) over multiple *generations*. Individuals are evaluated based on a *fitness function* (musical rules), and fitter individuals are more likely to be selected for *reproduction* (using *crossover* and *mutation*) to create the next generation.
2.  **Music Theory:** Principles governing musical structure, harmony, melody, and rhythm. The script incorporates rules related to scales, keys, diatonic chords, common chord progressions, melodic contour, and rhythmic patterns to guide the GA towards musically plausible results.
3.  **Audio Synthesis:** Simple techniques (sine waves, noise generation, ADSR envelopes) are used to convert the generated musical notation (MIDI pitches and durations) into audible waveforms.
4.  **Lofi Aesthetics:** Post-processing effects characteristic of the "lofi hip hop" genre are applied, such as filtering high frequencies, adding noise artifacts (vinyl/fire crackle), and simulating tape saturation.

## Configuration Parameters

Constants defined at the beginning of the script control the GA behavior and musical structure.

### Genetic Algorithm Parameters

*   `POPULATION_SIZE`: Number of melodies (individuals) in each generation (e.g., 100). Larger populations explore more possibilities but require more computation.
*   `NUM_GENERATIONS`: How many evolutionary cycles to run (e.g., 50). More generations allow for more refinement.
*   `TOURNAMENT_SIZE`: Number of individuals randomly selected to compete for selection (e.g., 5). A smaller part of the population competes based on fitness.
*   `CROSSOVER_RATE`: Probability (0.0 to 1.0) that two selected parents will exchange genetic material (parts of their melodies) (e.g., 0.8).
*   `MUTATION_RATE`: Initial probability (0.0 to 1.0) that a random change will occur in an individual's melody (e.g., 0.2). This rate can adapt during the evolution.

### Music Theory & Structure Parameters

*   `NUM_MEASURES`: The length of the generated melody in measures (e.g., 4).
*   `SAMPLE_RATE`: Audio quality in samples per second (e.g., 44100 Hz).
*   `BASE_TEMPO`: The duration of a single beat in seconds (e.g., 0.5 corresponds to 120 BPM).
*   `BASE_DURATIONS`: List of common rhythmic durations relative to the beat (e.g., `[0.25, 0.5, 1.0]` could represent 16th, 8th, and quarter notes if `BASE_TEMPO` is a quarter note).
*   `CREATIVE_DURATIONS`: Less common durations (e.g., `[0.75, 1.5]` could represent dotted 8th and dotted quarter notes).
*   `CREATIVE_PROB`: Probability of using a "creative" duration within a measure.
*   `BEAT_TOTAL`: Number of beats per measure (e.g., 4, representing 4/4 time).
*   `KEY`: The root note (tonic) of the scale, represented as a MIDI number (e.g., 60 = Middle C).
*   `SCALE_TYPE`: The type of scale used ("major" or "minor").
*   `SCALE_INTERVALS`: Defines the intervals (in semitones) from the key's root note for the specified `SCALE_TYPE`. This dictionary determines which notes are "in key".

## Genetic Algorithm Implementation

The GA evolves melodies represented as lists of measures.

### 1. Individual Representation

*   **Chromosome (Individual):** A complete melody, represented as a list of measures (`list[list[tuple[int, float]]]`). Example: `[[ (60, 0.5), (62, 0.5) ], [ (64, 1.0) ]]` represents a two-measure melody.
*   **Gene (Segment):** A single measure within the melody, represented as a list of `(pitch, duration)` tuples.
*   `create_individual()`: Generates a random individual (melody) by calling `generate_measure()` multiple times.
*   `generate_measure()`: Creates a single measure, randomly choosing notes from `ALLOWED_NOTES` and durations from `BASE_DURATIONS` (and sometimes `CREATIVE_DURATIONS`), attempting to fill the `BEAT_TOTAL`.

### 2. Fitness Function (`fitness`)

This function evaluates how "musically good" a generated melody is, assigning it a numerical score. It encodes basic music theory principles:

*   **Melodic Contour (Interval Size):** It iterates through consecutive notes, calculating the interval (distance in semitones).
    *   *Music Theory:* Prefers conjunct motion (steps) or small leaps. Large leaps often sound disjointed.
    *   *Implementation:* Rewards intervals of 5 semitones or less. Penalizes larger intervals increasingly (`score += 1 if interval <= 5 else - (interval - 5)`).
*   **Melodic Coherence (Interval Change):** It examines the *change* in size between successive intervals.
    *   *Music Theory:* Smooth melodies often maintain a consistent direction or type of movement. Abrupt changes from small steps to large leaps (or vice-versa) can be jarring.
    *   *Implementation:* Penalizes large differences between consecutive interval sizes (`if abs(intervals[i] - intervals[i-1]) > 3: score -= 1`).
*   **Tonal Resolution (Start/End Notes):** Checks if the first and last notes of the melody belong to the pitch class of the tonic (the `KEY`).
    *   *Music Theory:* Starting and ending on the tonic provides a strong sense of key and closure, fundamental in Western tonal music.
    *   *Implementation:* Adds points if the first note's pitch class matches the key's pitch class (`melody[0][0] % 12 == KEY % 12`) and similarly for the last note.

### 3. Selection (`tournament_selection`)

*   **Purpose:** To choose parent individuals for breeding the next generation, favoring fitter individuals without entirely discarding weaker ones.
*   **Method:** Tournament Selection.
    *   A small group (`TOURNAMENT_SIZE`) of individuals is randomly selected from the population.
    *   Their fitness scores are calculated.
    *   The individual with the highest fitness score in that tournament is chosen as a parent.

### 4. Crossover (`crossover`)

*   **Purpose:** To combine genetic material (musical ideas) from two parents to create offspring, potentially inheriting good traits from both.
*   **Method:** Single-Point Crossover (at measure boundaries).
    *   With probability `CROSSOVER_RATE`, a random point *between measures* is selected.
    *   The measures after this point are swapped between the two parents to create two new child melodies.
    *   If no crossover occurs (based on `CROSSOVER_RATE`), the children are simply copies of the parents.
    *   *Music Theory Rationale:* Crossing over at measure boundaries helps preserve potentially coherent musical phrases within measures.

### 5. Mutation (`mutate`)

*   **Purpose:** To introduce random variations into melodies, maintaining genetic diversity and preventing premature convergence to suboptimal solutions.
*   **Methods:** Applied with probability `MUTATION_RATE`:
    *   **Measure Replacement (50% chance):** An entire measure within the melody is replaced with a completely new, randomly generated measure using `generate_measure()`. This allows for larger structural changes.
    *   **Note Mutation (50% chance):** A single note within a randomly chosen measure is altered:
        *   Its pitch is changed to a random note from `ALLOWED_NOTES`.
        *   Its duration is changed to a random duration from the allowed lists.
        *   An attempt is made to adjust the *last* note's duration in the measure to compensate and maintain the correct total `BEAT_TOTAL` for the measure, preserving rhythmic integrity where possible.

### 6. Evolutionary Process (`run_ga`)

*   **Initialization:** Creates an initial population of random melodies.
*   **Generational Loop:** Repeats for `NUM_GENERATIONS`:
    1.  **Evaluation:** (Implicitly done during selection) Fitness of individuals is assessed.
    2.  **Selection:** Parents are chosen using `tournament_selection`.
    3.  **Reproduction:** Pairs of parents produce offspring using `crossover`.
    4.  **Mutation:** Offspring melodies are potentially modified by `mutate`.
    5.  **Population Replacement:** The new generation of offspring replaces the old population.
*   **Elitism (Implicit):** While not explicitly coded as keeping the single best, tournament selection naturally gives the best individuals a high chance of being selected multiple times, effectively carrying over good traits.
*   **Adaptive Mutation Rate:** If the best fitness score in the population hasn't improved for several generations (stagnation), the `MUTATION_RATE` is temporarily increased. This injects more diversity to help the algorithm escape local optima in the fitness landscape. The rate resets to the base `MUTATION_RATE` when fitness improves again.
*   **Output:** Returns the best melody (individual) found across all generations.

## Music Theory Implementation Details

Beyond the fitness function, music theory guides the generation process:

### 1. Scales and Keys (`get_allowed_notes`)

*   **Concept:** Diatonic scales (like major and minor) form the basis of tonal harmony. Melodies primarily using notes from the established key's scale sound consonant and coherent.
*   **Implementation:** `get_allowed_notes` calculates the set of allowed MIDI pitches based on the `KEY` and `SCALE_TYPE` (using `SCALE_INTERVALS`). The `generate_measure` function exclusively uses pitches from this pre-calculated set.

### 2. Rhythm and Measures (`generate_measure`)

*   **Concept:** Music is organized into measures with a consistent number of beats (`BEAT_TOTAL`). Rhythms are created using notes of varying durations.
*   **Implementation:** `generate_measure` attempts to fill a measure with `BEAT_TOTAL` beats using durations from `BASE_DURATIONS` and `CREATIVE_DURATIONS` (scaled by `BASE_TEMPO`). It ensures the last note fills the remaining duration, enforcing (approximately) the correct measure length.

### 3. Harmony: Chords and Progressions

*   **Concept:** Chords are built from scale notes (diatonic harmony). Chord progressions (sequences of chords) create the harmonic structure of a piece. Common progressions provide familiar and satisfying harmonic movement.
*   **`get_chord_from_scale`:** Builds a basic three-note chord (triad) based on a given scale degree (e.g., degree 0 = tonic chord, degree 4 = dominant chord if 0-indexed). It selects the root, third, and fifth notes *diatonically* from the `ALLOWED_NOTES` determined by the key and scale. Includes a common music theory adjustment: in minor keys, the third of the dominant chord (degree V, index 4) is often raised to create a major chord, strengthening its pull back to the tonic.
*   **`generate_chord_track`:** Creates the chord accompaniment. It uses a predefined, common chord progression (`progression_degrees = [0, 5, 3, 4]`, which might represent i-VI-iv-V in C minor, or I-vi-IV-V in C major). It synthesizes each chord in the sequence for a fixed duration (`chord_duration`) and repeats the progression to match the melody's length.

## Audio Synthesis

Functions convert the abstract musical data into sound:

*   `midi_to_freq`: Converts MIDI note numbers to frequencies (Hz).
*   `generate_adsr_envelope`: Creates an Attack-Decay-Sustain-Release envelope to shape the volume of synthesized notes over time, making them sound less static.
*   `synthesize_note`: Generates a sine wave at the target frequency and applies the ADSR envelope.
*   `synthesize_melody`: Concatenates synthesized notes to create the full melody audio.
*   `synthesize_chord`: Generates chord audio by summing the sine waves of the constituent notes and applying an envelope.
*   `synthesize_kick`, `synthesize_snare`, `synthesize_hihat`, `synthesize_bass`: Use basic synthesis techniques (frequency sweeps, noise, simple waveforms with envelopes) to approximate drum and bass sounds.

## Rhythm Section Generation

*   `create_drum_loop`: Arranges synthesized drum sounds (`kick`, `snare`, `hihat`) according to a `measure_config` list, which specifies the beat offset and instrument for each hit within a repeating measure.
*   `drum_pattern_X` functions (`drum_pattern_1` to `drum_pattern_5`): Define specific `measure_config` lists corresponding to common drum grooves (e.g., four-on-the-floor, basic backbeat). They also generate a simple bassline, often playing the root note of the current chord or a prominent melody note on strong beats.
*   `generate_random_drum_track`: Randomly selects one of the predefined `drum_pattern_X` functions.

## Lofi Effects

Applied after mixing to achieve the desired aesthetic:

*   `low_pass_filter`: Removes high frequencies using a simple digital filter. This is a hallmark of the lofi sound, giving it a "muffled" or "warm" quality. The cutoff frequency can be randomized slightly.
*   `apply_tape_saturation`: Simulates the effect of overloading analog tape using the `tanh` function. This introduces subtle compression and harmonic distortion, often perceived as "warmth".
*   `generate_lofi_artifact` / `add_lofi_artifacts`: Generates and adds low-level noise simulating vinyl crackle or fireplace sounds, another common lofi texture.

## Mixing and Output

*   `mix_tracks`: Combines the synthesized melody, chord, and drum/bass audio tracks into a single stereo track. Relative volumes are adjustable. Tracks are normalized before and after mixing to prevent clipping.
*   `play_audio`: Uses the `sounddevice` library to play back the final audio array.
*   `wav_bytes`: Converts the final audio (NumPy array) into WAV file format (as bytes), suitable for saving.

## Main Orchestration (`generate_music`)

This function ties everything together:

1.  Runs the GA (`run_ga`) to get the `best_individual` melody.
2.  Optionally transposes the melody and key to fit a desired pitch range or for variation.
3.  Synthesizes the melody audio (`synthesize_melody`).
4.  Determines the total duration based on the synthesized melody.
5.  Generates the chord track (`generate_chord_track`) matching the duration and (transposed) key.
6.  Generates the drum/bass track (`generate_random_drum_track` or a specific pattern) for the same duration.
7.  Mixes the tracks (`mix_tracks`).
8.  Applies selected lofi effects (artifacts, saturation, low-pass filter).
9.  Normalizes the final output.
10. Returns the final audio array and the sample rate.

## Usage Example

The `if __name__ == "__main__":` block demonstrates how to use the `generate_music` function:

*   Set configuration flags (e.g., `APPLY_TAPE_EFFECT`, `NUMBER_OF_MEASURES`).
*   Call `generate_music` with the desired settings.
*   Play the resulting audio using `play_audio`.
*   Save the audio to a file (`lofi_output.wav`) using `wav_bytes`.
