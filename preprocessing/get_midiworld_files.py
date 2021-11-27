"""Downloads midi files from midiworld given a genre and saves them
You must manually create a folder named whatever your GENRE is before running"""

from bs4 import BeautifulSoup
import requests

# CHANGE THESE
# number of pages midiworld has in this genre
NUM_PAGES = 2
GENRE = "jazz"

# ================================ website info ================================
MIDI_BASE_URL = "https://www.midiworld.com/search/"
GENRE_URL = "/?q=" + GENRE
# titles all had - download in them
SPLIT_TEXT = " - download"

# ============================= file/directory info ============================
DIRECTORY = "data/" + GENRE + "/"
FILE_EXTENSION = ".mid"

# ==================================== code ====================================
for n in range(1, NUM_PAGES + 1):
    print(n)
    r = requests.get(MIDI_BASE_URL + str(n) + GENRE_URL)
    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.find_all("ul")[1].find_all("li")
    for link in links:
        file_name = (
            link.text.strip().split(SPLIT_TEXT)[0].replace(" ", "_").replace("/", "_")
            + FILE_EXTENSION
        )
        print(file_name)
        music_file = requests.get(link.a.get("href"))
        open(DIRECTORY + file_name, "wb").write(music_file.content)
