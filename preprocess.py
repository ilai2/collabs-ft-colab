from os import startfile
import tensorflow as tf
import numpy as np
from music21 import *
from fractions import Fraction
import glob
# Hyperparameters
INSTRUMENT_NUM = 18

# Global Variables

# element name to int dictionary
# notes are represented as strings in the form "n: <pitch>"
# chords are represented in the form "c: <pitch_1>.<pitch_2>. etc."
# rests are "r"
# padding is "p"
element_to_int_dict = {"p": 0}
# instrument_class to integer dictionary
instrument_to_int_dict = {
    type(instrument.Instrument()): 0,
    type(instrument.WoodwindInstrument()): 1,
    type(instrument.BrassInstrument()): 2,
    type(instrument.ElectricOrgan()): 3,
    type(instrument.Choir()): 4,
    type(instrument.FretlessBass()): 5,
    type(instrument.ElectricPiano()): 6,
    type(instrument.Vocalist()): 7,
    type(instrument.Bass()): 8,
    type(instrument.ElectricBass()): 9,
    type(instrument.ElectricGuitar()): 10,
    type(instrument.Sampler()): 11,
    type(instrument.StringInstrument()): 12,
    type(instrument.Piano()): 13,
    type(instrument.UnpitchedPercussion()): 14,
    type(instrument.AcousticGuitar()): 15,
    type(instrument.AltoSaxophone()): 16,
    type(instrument.KeyboardInstrument()): 17
}
# the same as element_to_int_dict but with keys and values swapped
int_to_element_dict = {0: "p"}

# the same as instrument_to_int_dict but with keys and values swapped
int_to_instrument_dict = {
    0: instrument.Instrument(),
    1: instrument.Flute(),
    2: instrument.Trumpet(),
    3: instrument.ElectricOrgan(),
    4: instrument.Choir(),
    5: instrument.FretlessBass(),
    6: instrument.ElectricPiano(),
    7: instrument.Vocalist(),
    8: instrument.Bass(),
    9: instrument.ElectricBass(),
    10: instrument.ElectricGuitar(),
    11: instrument.Sampler(),
    12: instrument.Guitar(),
    13: instrument.Piano(),
    14: instrument.UnpitchedPercussion(),
    15: instrument.AcousticGuitar(),
    16: instrument.AltoSaxophone(),
    17: instrument.Piano()
}

def process_instrument(instr):
    """ 
    param: an instrument
    returns: the integer associated with that instrument in instrument_to_int_dict
    if the instrument does not exist in the dictionary, it returns None
    """
    instrumentType = type(instr)
    try:
        return instrument_to_int_dict[instrumentType]
    except:
        if issubclass(instrumentType, type(instrument.StringInstrument())):
            return instrument_to_int_dict[type(instrument.StringInstrument())]
        elif issubclass(instrumentType, type(instrument.KeyboardInstrument())):
            return instrument_to_int_dict[type(instrument.KeyboardInstrument())]
        elif issubclass(instrumentType, type(instrument.WoodwindInstrument())):
            return instrument_to_int_dict[type(instrument.WoodwindInstrument())]
        elif issubclass(instrumentType, type(instrument.BrassInstrument())):
            return instrument_to_int_dict[type(instrument.BrassInstrument())]
        elif issubclass(instrumentType, type(instrument.Vocalist())):
            return instrument_to_int_dict[type(instrument.Vocalist())]
        else:
            return None
def process_element(element, old_element):
    """ 
    param: a note, chord or rest, and an element that occurs at the same time
    returns: the integer associated with that element in element_to_int_dict
    if the element does not exist in the dictionary, it adds it first
    """
    old_string = int_to_element_dict[old_element]
    if old_string[0] == 'n' or old_string[0] == 'c':
        start = "c:" + old_string[2:] + "."
    elif isinstance(element, chord.Chord):
        start = "c:"
    elif isinstance(element, note.Note):
        start = "n:"
    
    if isinstance(element, note.Note):
        element_string = start + str(element.pitch)
    elif isinstance(element, chord.Chord):
        element_string = start + '.'.join(str(n) for n in element.pitches)
    elif isinstance(element, note.Rest):
        element_string = "r"
    
    if element_string in element_to_int_dict:
        encoding = element_to_int_dict[element_string]
    else:
        encoding = len(element_to_int_dict)

        element_to_int_dict[element_string] = encoding
        int_to_element_dict[encoding] = element_string
    return encoding

def process_part(notes_to_parse, instrument_encoding, notes, durations, volumes):
    """ 
    param: a music21 stream, called notes_to_parse, the instrument that the part is for, and 3 instrument_amount * time matrices 
    returns: a tuple of 3 lists
        notes: a matrix of the notes (represented as integers) in the stream. At index (i,j), there is an encoding
        of the note of the ith instrument at the offset j/12. 0 represents a lack of note at an offset
        durations: a parallel matrix of durations, each duration is either a Fraction or float
        and represents the duration of the note in the same index
        volumes: a parallel matrix of volumes, each volume is an integer
        and represents the volume of the note in the same index
    """
    for element in notes_to_parse:
        if (isinstance(element, note.Note)
        or isinstance(element, chord.Chord) 
        or isinstance(element, note.Rest)):
            index =  int(round(float(element.offset) * 12))
            notes[instrument_encoding][index] = process_element(element, notes[instrument_encoding][index])
            durations[instrument_encoding][index] = int(element.duration.quarterLength*12)
            if not (isinstance(element, note.Rest)):
                volumes[instrument_encoding][index] = element.volume.velocity
    return (notes, durations, volumes)

def offset_length(part):
    """Finds the largest offset of a element in a part"""
    try:
        return part[-1].offset
    except: 
        return 0

def process_midi(filepath):
    """ 
    param: a filepath to a .mid file
    returns: a tuple of 3 matrices
        the 3 elements of the tuple are:
            the notes matrix
            the durations matrix
            the volumes matrix
    """
    midi = converter.parse(filepath)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts: # file has instrument parts
        maximum_offset = max([offset_length(part) for part in parts])
        size = int(round(float(maximum_offset) * 12)) + 1
        notes = np.zeros((INSTRUMENT_NUM, size))
        durations = np.zeros((INSTRUMENT_NUM, size))
        volumes = np.zeros((INSTRUMENT_NUM, size))
        for part in parts:
            instrument_encoding = process_instrument(part.getInstrument())
            if instrument_encoding != None:
                notes_to_parse = part.recurse()
                notes, durations, volumes = process_part(notes_to_parse, instrument_encoding, notes, durations, volumes)
        return (notes, durations, volumes)
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notesAndRests
        maximum_offset = max(notes_to_parse)
        size = int(round(float(maximum_offset) * 12)) + 1
        notes = np.zeros((INSTRUMENT_NUM, size))
        durations = np.zeros((INSTRUMENT_NUM, size))
        volumes = np.zeros((INSTRUMENT_NUM, size))
        return process_part(notes_to_parse, 0, notes, durations, volumes)

def write_element_dict(filepath):
    with open(filepath, "w+") as f:
        f.write(str(len(element_to_int_dict)) + "\n")
        for key in element_to_int_dict.keys():
            f.write(str(element_to_int_dict[key]) + " " + key + "\n")
    pass

def write_song(filepath, notes, durations, volumes):
    (xArray, yArray) = np.nonzero(notes)
    with open(filepath, "a+") as f:
        f.write(str(len(notes[0])))
        for i in range(len(xArray)):
            x = xArray[i]
            y = yArray[i]
            f.write(";")
            f.write(str(x))
            f.write(",")
            f.write(str(y))
            f.write(",")
            f.write(str(notes[x][y]))
            f.write(",")
            f.write(str(durations[x][y]))
            f.write(",")
            f.write(str(volumes[x][y]))
        f.write("\n")
    pass

def read_element_dict(filepath):
    with open(filepath, "r+") as f:
        dict = {}
        size = int(f.readline())
        line = f.readline()
        while line:
            split = line.split()
            dict[int(split[0])] = split[1]
            line = f.readline()
    return (size, dict)

def read_song(filepath,lineno):
    with open(filepath, "r+") as f:
        for (i, line) in enumerate(f):
            if i == lineno:
                line = line.split(";")
                notes = np.zeros((INSTRUMENT_NUM, int(line[0])))
                durations = np.zeros((INSTRUMENT_NUM, int(line[0])))
                volumes = np.zeros((INSTRUMENT_NUM, int(line[0])))
                for tuple in line[1:]:
                    split = tuple.split(",")
                    x = int(split[0])
                    y = int(split[1])
                    notes[x][y] = int(float(split[2]))
                    durations[x][y] = int(float(split[3]))
                    volumes[x][y] = int(float(split[4]))
                break
    return (notes, durations, volumes)



def deprocess_midi(notes, durations, volumes):
    """
    param :
            the notes matrix
            the durations matrix
            the volumes matrix
    returns : a midi stream of the encoding
    """
    midi_score = stream.Score()
    for i in range(len(notes)):
        p = stream.Part()
        p.insert(0, int_to_instrument_dict[i])
        p.insert(0, tempo.MetronomeMark(number = 120))
        p.insert(0, meter.TimeSignature('4/4'))
        offset = 0
        for j in range(len(notes[i])):
            element_encoding = notes[i][j]
            element_string = int_to_element_dict[element_encoding]
            if (element_string[0] == 'n'):
                new_note = note.Note(element_string[2:])
                new_duration = duration.Duration()
                new_duration.quarterLength = Fraction(int(durations[i][j]), 12)
                new_note.duration = new_duration
                new_note.volume = volumes[i][j]
                p.insert(Fraction(offset, 12), new_note)
            elif (element_string[0] == 'r'):
                new_rest = note.Rest()
                new_duration = duration.Duration()
                new_duration.quarterLength = Fraction(int(durations[i][j]), 12)
                new_rest.duration = new_duration
                p.insert(Fraction(offset, 12), new_rest)
            elif (element_string[0] == 'c'):
                notes_in_chord = element_string[2:].split('.')
                chordNotes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(current_note)
                    chordNotes.append(new_note)
                new_chord = chord.Chord(chordNotes)
                new_duration = duration.Duration()
                new_duration.quarterLength = Fraction(int(durations[i][j]), 12)
                new_chord.duration = new_duration
                new_chord.volume = volumes[i][j]
                p.insert(Fraction(offset, 12), new_chord)
            offset = offset + 1
        midi_score.insert(0, p)
    return midi_score

def main():
    folder_name = "data_collection/freemidi_data/freemidi_data/alternative-indie/"
    
    start_file = 0
    count = 0

    global element_to_int_dict
    try: 
        element_to_int_dict = read_element_dict(folder_name + "dict.txt")
    except:
        element_to_int_dict = {}
    
    for file in glob.glob(folder_name + "*.mid"):
        if (count >= start_file):
            try:
                (notes, durations, volumes) = process_midi(file)
                write_song(folder_name + "songs.txt", notes, durations, volumes)
            except:
                pass 
            if count > 500:
                print(count)
                break
        count = count + 1

    write_element_dict(folder_name + "dict.txt")

if __name__ == '__main__':
	main()