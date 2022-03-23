#!/usr/bin/env python3

import sys
import time
import os

from os import listdir
from os.path import isfile, join

from pytube import YouTube
import eyed3


def getVideoUrls():
    videos = []

    f = open("data/youtubeURL_LL.txt", "r")
    for x in f:
        videos.append(x.strip('\n'))

    if videos:
        print("Found", len(videos), "videos in playlist.")
        return videos
    else:
        print('No videos found.')
        exit(1)


# function added to get audio files along with the video files from the playlist
def download_Video(path, vid_url):
    try:
        yt = YouTube(vid_url)
    except Exception as e:
        print("Error:", str(e), "- Skipping Video '" + vid_url + "'.")
        return

    try:
        video = yt.streams.filter(adaptive=True, only_audio=True).first()
    except Exception as e:
        print("Error:", str(e), "- Skipping Video with title '" + yt.title + "'.")
        return

    try:
        video.download(path)
        print("Successfully downloaded", yt.title)
    except OSError:
        print(yt.title, "already exists in this directory! Skipping video...")


def transform_Audio(dir):
    files = [os.path.splitext(f)[0] for f in listdir(dir) if isfile(join(dir, f))]
    for i, f in enumerate(files):
        print("Transforming to audio ", f)
        try:
            vid = dir + '/' + f + '.mp4'
            id_vid = dir + '/' + str(i) + '.mp4'
            aud = dir + '/' + f + '.mp3'

            os.rename(vid, id_vid)
            os.system('ffmpeg -loglevel quiet -i "' + id_vid + '" "' + aud + '"')
            os.remove(id_vid)

            is_remix = "remix" in f.casefold() or "mashup" in f.casefold() or "bootleg" in f.casefold()

            audiofile = eyed3.load(aud)
            audiofile.tag.artist = f.split('-')[0]
            audiofile.tag.title = f.split('-')[1]
            audiofile.tag.genre = "Other" if is_remix else "Dance"
            audiofile.tag.save()
        except Exception:
            print("\tError with the transformation of", f)


if __name__ == '__main__':
    if len(sys.argv) < 1 or len(sys.argv) > 2:
        print('USAGE: python ytPlaylistDL.py destPath')
        exit(1)
    else:
        directory = os.getcwd() if len(sys.argv) != 2 else sys.argv[1]

        # make directory if dir specified doesn't exist
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(e.reason)
            exit(1)

        vid_urls = getVideoUrls()

        for vid_name_url in vid_urls:
            download_Video(directory, vid_name_url)
            time.sleep(1)

        transform_Audio(directory)
