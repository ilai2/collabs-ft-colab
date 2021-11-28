
import tensorflow as tf
import numpy as np
from music21 import *
from fractions import Fraction

# Hyperparameters
SEQUENCE_LENGTH = 10

# Global Variables

# element name to int dictionary
# notes are represented as strings in the form "n: <pitch>"
# chords are represented in the form "c: <pitch_1>.<pitch_2>. etc."
# rests are "r"
# padding is "p"
element_to_int_dict = {"p": 0}
# instrument to instrument name dictionary
instrument_to_int_dict = {}
# the same as element_to_int_dict but with keys and values swapped
int_to_element_dict = {0: "p"}
# the same as instrument_to_int_dict but with keys and values swapped
int_to_instrument_dict = {}

def process_instrument(instrument):
    """ 
    param: an instrument
    returns: the integer associated with that instrument in instrument_to_int_dict
    if the instrument does not exist in the dictionary, it adds it first
    """
    if instrument in instrument_to_int_dict:
        instrument_encoding = instrument_to_int_dict[instrument]
    else:
        instrument_to_int_dict[instrument] = len(instrument_to_int_dict)
        int_to_instrument_dict[len(instrument_to_int_dict)] = instrument
        instrument_encoding = len(instrument_to_int_dict)
    return instrument_encoding

def process_element(element):
    """ 
    param: a note, chord or rest
    returns: the integer associated with that element in element_to_int_dict
    if the element does not exist in the dictionary, it adds it first
    """
    if isinstance(element, note.Note):
        element_string = "n:" + str(element.pitch)
    elif isinstance(element, chord.Chord):
        element_string = "c:" + '.'.join(str(n) for n in element.pitches)
    elif isinstance(element, note.Rest):
        element_string = "r"
    
    if element_string in element_to_int_dict:
        encoding = element_to_int_dict[element_string]
    else:
        encoding = len(element_to_int_dict)

        element_to_int_dict[element_string] = encoding
        int_to_element_dict[encoding] = element_string
    return encoding

def process_part(notes_to_parse):
    """ 
    param: a music21 stream, called notes_to_parse
    returns: a tuple of two lists
        notes: a list of the notes (represented as integers) in the stream. At index i, there is an encoding
        of the note at offset i/12. 0 represents a lack of note
        durations: a parallel list of durations, each duration is either a Fraction or float
        and represents the duration of the note in the same index
    """

    size = int(round(float(notes_to_parse[-1].offset) * 12)) + 1
    notes = [0] * size
    durations = [0] * size
    volumes = [0] * size
    for element in notes_to_parse:
        if (isinstance(element, note.Note)
        or isinstance(element, chord.Chord) 
        or isinstance(element, note.Rest)):
            index =  int(round(float(element.offset) * 12))
            notes[index] = process_element(element)
            durations[index] = float(element.duration.quarterLength)
            if not (isinstance(element, note.Rest)):
                volumes[index] = element.volume.velocity
    return (notes, durations, volumes)

def process_midi(filepath):
    """ 
    param: a filepath to a .mid file
    returns: a list of 3-tuples
        each element represents the part of one instrument
        the 3 elements of the tuple are:
            the instrument of the part (as an int)
            the notes of the part (as a list of ints)
            the duration of the part (as a list of Fractions and floats)
    """
    #TODO: make the 
    midi = converter.parse(filepath)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts: # file has instrument parts
        score = []
        index = 0
        for part in parts:
            instrument_encoding = process_instrument(part.getInstrument())
            notes_to_parse = part.recurse()
            (notes, durations, volumes) = process_part(notes_to_parse)
            score.append((instrument_encoding, notes, durations, volumes))
            index = index + 1
        return score
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notesAndRests
        instrument_encoding = process_instrument(parts.getInstrument())
        (notes, durations, volumes) = process_part(notes_to_parse)
        return [(instrument_encoding, notes, durations, volumes)]

def deprocess_midi(encoding):
    midi_score = stream.Score()
    for part in encoding:
        p = stream.Part()
        p.insert(0, int_to_instrument_dict[part[0]])
        p.insert(0, tempo.MetronomeMark(number = 120))
        p.insert(0, meter.TimeSignature('4/4'))
        offset = 0
        for i in range(len(part[1])):
            element_encoding = part[1][i]
            element_string = int_to_element_dict[element_encoding]
            if (element_string[0] == 'n'):
                new_note = note.Note(element_string[2:])
                new_duration = duration.Duration()
                new_duration.quarterLength = part[2][i]
                new_note.duration = new_duration
                new_note.volume = part[3][i]
                p.insert(Fraction(offset, 12), new_note)
            elif (element_string[0] == 'r'):
                new_rest = note.Rest()
                new_duration = duration.Duration()
                new_duration.quarterLength = part[2][i]
                new_rest.duration = new_duration
                p.insert(Fraction(offset, 12), new_rest)
            elif (element_string[0] == 'c'):
                notes_in_chord = element_string[2:].split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(current_note)
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_duration = duration.Duration()
                new_duration.quarterLength = part[2][i]
                new_chord.duration = new_duration
                new_chord.volume = part[3][i]
                p.insert(Fraction(offset, 12), new_chord)
            offset = offset + 1
        midi_score.insert(0, p)
    return midi_score

def main():
    midi = process_midi("midis/red.mid")
    midi_stream = deprocess_midi(midi)
    midi_stream.write('midi', fp='test_output.mid')
if __name__ == '__main__':
	main()