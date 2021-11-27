
import tensorflow as tf
import numpy as np
from music21 import converter, instrument, note, chord

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
        instrument_encoding = instrument_to_int_dict[instrument.instrumentName]
    else:
        instrument_to_int_dict[instrument.instrumentName] = len(instrument_to_int_dict)
        int_to_instrument_dict[len(instrument_to_int_dict)] = instrument.instrumentName
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
    notes = []
    durations = []
    currOffset = 0
    for element in notes_to_parse:
        if (isinstance(element, note.Note)
        or isinstance(element, chord.Chord) 
        or isinstance(element, note.Rest)):
            print(element.offset)
            if(element.offset != currOffset):
                notes.extend([0] * int((float(element.offset) - currOffset) * 12))
            notes.append(process_element(element))
            durations.append(float(element.duration.quarterLength))
            currOffset = element.offset + 1/12
    print(notes)
    return (notes, durations)

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
            (notes, durations) = process_part(notes_to_parse)
            score.append((instrument_encoding, notes, durations))
            print(len(notes_to_parse))
            index = index + 1
        return score
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
        instrument_encoding = process_instrument(parts.getInstrument())
        (notes, durations) = process_part(notes_to_parse)
        return [(instrument_encoding, notes, durations)]
        
def main():
    midi = process_midi("midis/Red.mid")
    print(midi)

if __name__ == '__main__':
	main()