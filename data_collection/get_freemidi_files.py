"""Downloads midi files from freemidi and sorts them by genre
You must manually create genre folders before running the script"""

import os
from bs4 import BeautifulSoup
import requests

BASE_URL = "https://freemidi.org"
DIRECTORY = "data/"
LETTERS_TO_PAGES = {
    "0": 9,
    "a": 55,
    "b": 62,
    "c": 51,
    "d": 50,
    "e": 24,
    "f": 35,
    "g": 31,
    "h": 45,
    "i": 60,
    "j": 15,
    "k": 12,
    "l": 51,
    "m": 49,
    "n": 28,
    "o": 26,
    "p": 35,
    "q": 2,
    "r": 34,
    "s": 98,
    "t": 86,
    "u": 10,
    "v": 8,
    "w": 50,
    "x": 1,
    "y": 16,
    "z": 2,
}


def download_song(url, folders, file_name):
    """Downloads a song and saves it to appropriate folders for its genre

    inputs:
        url - the url for the song page we want to download
        folders - a list of strings which are the song's tagged genres
        file_name - the name of the file to save the midi to, of the form
                    SONG_TITLE_(ARTIST_NAME)
    """
    try:
        r = requests.get(url + "?PageSpeed=noscript")
        soup = BeautifulSoup(r.text, "html.parser")
        # obtain the download link from song page link
        getter_end_url = soup.find("a", {"id": "downloadmidi"}).get("href")

        # NOTE: for some reason, I could only get this download to work with
        #       cURL, so I didn't use requests.get here. the error messages are
        #       nicer anyway :D
        # download the song to a temporary file, suppress non-error messages
        os.system(f"curl -s -S '{BASE_URL}/{getter_end_url}' > midway")
        # copy the midi file to each genre folder that the song is in
        for folder in folders:
            if folder != "":
                os.system(f'cp midway "{DIRECTORY}{folder}/{file_name}.mid"')

    except requests.exceptions.TooManyRedirects as err:
        print("oops, too many redirects for " + url)
    except requests.exceptions.ConnectionError as err:
        print(err)


# go through all songs by beginning letter of song title
for letter in LETTERS_TO_PAGES:
    print("=====================================")
    for page in range(LETTERS_TO_PAGES[letter]):
        print(f"letter {letter}, page {page+1}")
        # using the noscript query parameter at the end loads the full html
        r = requests.get(f"{BASE_URL}/songtitle-{letter}-{page}?PageSpeed=noscript")
        soup = BeautifulSoup(r.text, "html.parser")

        # extract list of songs from the html
        songs = soup.find_all("div", {"class": "song-list-container"})
        for song in songs:
            song_info = song.find("div", {"class": "row-title"}).a
            # get song title and artist and replace spaces with _
            title = song_info.get("title").strip().replace(" ", "_")
            artist = (
                song.find("div", {"class": "row-directory"})
                .a.text.strip()
                .replace(" ", "_")
            )
            # create file name, safe-ify it for later by replacing " with ' and
            # / with _ (so it can be used in the cURL command later)
            file_name = f"{title}_({artist})".replace('"', "'").replace("/", "_")
            # form song page url
            link = BASE_URL + song_info.get("href")
            genres = [
                x.text for x in song.find("div", {"class": "row-genre"}).find_all("a")
            ]
            # there is always an empty genre at the end of each row-genre div
            if len(genres) > 1:
                download_song(link, genres[:-1], file_name)
            elif artist == "Christmas":
                download_song(link, ["christmas"], file_name)
            elif artist == "National_Anthems":
                download_song(link, ["national-anthems"], file_name)
