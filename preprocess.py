import tensorflow as tf
import numpy as np
from music21 import *
from fractions import Fraction
import glob
# Hyperparameters
INSTRUMENT_NUM = 18
VOCAB_SIZE = 5000
# Global Variables

# element name to int dictionary
# elements are represented as a set of pitches (integers), keys are actually strings b/c no mutable objects as keys
element_to_int_dict = {"set()" : 0}
# instrument_class to integer dictionary
element_to_occurences_dict = {"set()": 0}

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
int_to_element_dict = {0: set()}

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
    old_set = int_to_element_dict[old_element]
    if isinstance(element, note.Note):
        element_set = {element.pitch.midi}
    elif isinstance(element, chord.Chord):
        element_set = {p.midi for p in element.pitches}
    else:
         element_set = set()
    element_set.union(old_set)
    if str(element_set) in element_to_int_dict:
        # element_to_occurences_dict[str(element_set)] = element_to_occurences_dict[str(element_set)] + 1
        encoding = element_to_int_dict[str(element_set)]
    else:
        encoding = 0
        # encoding = len(element_to_int_dict)
        # element_to_occurences_dict[str(element_set)] = 1
        # element_to_int_dict[str(element_set)] = encoding
        # int_to_element_dict[encoding] = element_set
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
            f.write(str(element_to_int_dict[key]) + " " + str(key) + "\n")
    pass

def write_occurences_dict(filepath):
    with open(filepath, "w+") as f:
        f.write(str(len(element_to_occurences_dict)) + "\n")
        for key in element_to_occurences_dict.keys():
            f.write(str(element_to_occurences_dict[key]) + " " + str(key) + "\n")
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
        dict = {"set()": 0}
        size = int(f.readline())
        f.readline()
        line = f.readline()
        while line:
            split = line.split("{")
            dict["{" + split[1].strip()] = int(split[0].strip())
            line = f.readline()
    return (size, dict)

def stringToSet(setString):
    setString = setString[1:-1]
    newSet = set()
    for x in setString.split(","):
       newSet.add(int(x.strip()))
    return newSet

def read_int_dict(filepath):
    with open(filepath, "r+") as f:
        dict = {0: set()}
        size = int(f.readline())
        f.readline()
        line = f.readline()
        while line:
            split = line.split("{")
            dict[int(split[0].strip())] = stringToSet("{" + split[1][:-1])
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
                    if int(float(split[2])) < VOCAB_SIZE:
                        notes[x][y] = int(float(split[2]))
                        durations[x][y] = int(float(split[3]))
                        volumes[x][y] = int(float(split[4]))
                break
    return (notes, durations, volumes)



def deprocess_midi(notes, durations, volumes, int_to_e_dict):
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
        p.insert(0, int_to_instrument_dict[i+16])
        p.insert(0, tempo.MetronomeMark(number = 120))
        p.insert(0, meter.TimeSignature('4/4'))
        offset = 0
        for j in range(len(notes[i])):
            element_encoding = int(notes[i][j])
            element_set = int_to_e_dict[element_encoding]
            if (len(element_set) == 1):
                new_note = note.Note(midi = list(element_set)[0])
                new_duration = duration.Duration()
                new_duration.quarterLength = Fraction(int(durations[i][j]), 12)
                new_note.duration = new_duration
                new_note.volume = volumes[i][j]
                p.insert(Fraction(offset, 12), new_note)
            elif (len(element_set) > 1):
                chordNotes = []
                for current_note in element_set:
                    new_note = note.Note(midi = current_note)
                    chordNotes.append(new_note)
                new_chord = chord.Chord(chordNotes)
                new_duration = duration.Duration()
                new_duration.quarterLength = Fraction(int(durations[i][j]), 12)
                new_chord.duration = new_duration
                new_chord.volume = volumes[i][j]
                p.insert(Fraction(offset, 12), new_chord)
            #offset = offset + 1
            offset = offset + 6
        midi_score.insert(0, p)
    return midi_score


def make_final_dict(read_file, write_file):
    with open(read_file, "r+") as f:    
        lines = f.readlines()[1:]
        def key_func(str):
            return int(str.split(" ")[0])
        lines.sort(reverse = True, key = key_func)
        with open(write_file, "w+") as g:
            count = 1
            line = lines[1]
            while key_func(line) >= 10:
                g.write(str(count))
                g.write(line[line.find(" "):]) 
                count = count + 1
                line = lines[count]

        
def main():
    # Change this to whatever your folder is!
    folder_name = "data_collection/freemidi_data/freemidi_data/metal/"
    
    # Change this to whatever the index of the last file you ran it on was!
    start_file = 0
    end_file = 0

    count = 0
    global element_to_int_dict
    global int_to_element_dict
    _, int_to_element_dict = read_int_dict("dict.txt")
    (_, element_to_int_dict) = read_element_dict("dict.txt")
    
    for file in glob.glob(folder_name + "*.mid"):
        if (count >= start_file):
            try:
                (notes, durations, volumes) = process_midi(file)
                write_song(folder_name + "songs.txt", notes, durations, volumes)
            except:
                pass
            if count >= end_file:
                print(count)
                break
        count = count + 1

if __name__ == '__main__':
	main()
