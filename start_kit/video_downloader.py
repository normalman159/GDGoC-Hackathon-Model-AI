import os
import json
import time
import sys
import urllib.request
from multiprocessing.dummy import Pool
import threading
import random

import logging
logging.basicConfig(filename='download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Set this to youtube-dl if you want to use youtube-dl.
# The the README for an explanation regarding yt-dlp vs youtube-dl.
youtube_downloader = "yt-dlp"

def request_video(url, referer=''):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    headers = {'User-Agent': user_agent,
               }
    
    if referer:
        headers['Referer'] = referer

    request = urllib.request.Request(url, None, headers)  # The assembled request

    logging.info('Requesting {}'.format(url))
    try:
        response = urllib.request.urlopen(request, timeout=8)  # Set timeout to 8 seconds
        data = response.read() # if successful, read the data
        return data
    except urllib.error.HTTPError as e:
        logging.error(f"HTTP Error {e.code} - {url}: {e.reason}")
    except urllib.error.URLError as e:
        logging.error(f"URL Error - {url}: {e.reason}")
    except Exception as e:
        logging.error(f"Unexpected error for {url}: {str(e)}")

    return None  # Return None if request fails


def save_video(data, saveto):
    with open(saveto, 'wb+') as f:
        f.write(data)

    # please be nice to the host - take pauses and avoid spamming
    time.sleep(random.uniform(0.5, 0.8))


def download_youtube(url, dirname, video_id):
    raise NotImplementedError("Urllib cannot deal with YouTube links.")


def download_aslpro(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.swf'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return 

    data = request_video(url, referer='http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi')

    if data is None:
        logging.warning(f"Skipping {video_id} due to inaccessible URL")
        return

    save_video(data, saveto)


def download_others(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.mp4'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto)) 
        return 
    
    data = request_video(url)
    if data is None: 
        logging.warning(f"Skipping {video_id} due to inaccessible URL")
        return

    save_video(data, saveto)


def select_download_method(url):
    if 'aslpro' in url:
        return download_aslpro
    elif 'youtube' in url or 'youtu.be' in url:
        return download_youtube
    else:
        return download_others


def download_nonyt_videos(inst, saveto='raw_videos'):
            video_url = inst['url']
            video_id = inst['video_id']
            
            logging.info('video: {}.'.format(video_id))

            download_method = select_download_method(video_url)    
            
            if download_method == download_youtube:
                logging.warning('Skipping YouTube video {}'.format(video_id))
                return

            try:
                download_method(video_url, saveto, video_id)
            except Exception as e:
                logging.error('Unsuccessful downloading - video {}'.format(video_id))

def download_nonyt_videos_multiple(saveto, MAX_THREADS=12) :
    content = json.load(open('WLASL_v0.3.json'))
    if not os.path.exists(saveto):
        os.mkdir(saveto)

    # Extract all YouTube video instances
    yt_instances = [
        inst for entry in content for inst in entry['instances']
        if 'youtube' not in inst['url'] and 'youtu.be' not in inst['url']
    ]
    with Pool(MAX_THREADS) as pool:
        pool.starmap(download_nonyt_videos, [(inst, saveto) for inst in yt_instances])

def check_youtube_dl_version():
    ver = os.popen(f'{youtube_downloader} --version').read()

    assert ver, f"{youtube_downloader} cannot be found in PATH. Please verify your installation."


def download_yt_videos(inst, saveto='raw_videos'):
    video_url = inst['url']
    video_id = inst['video_id']

    if 'youtube' not in video_url and 'youtu.be' not in video_url:
        return

    if os.path.exists(os.path.join(saveto, video_url[-11:] + '.mp4')) or os.path.exists(os.path.join(saveto, video_url[-11:] + '.mkv')):
        logging.info('YouTube videos {} already exists.'.format(video_url))
        return
    
    else:
        cmd = f"{youtube_downloader} \"{{}}\" -o \"{{}}%(id)s.%(ext)s\""
        cmd = cmd.format(video_url, saveto + os.path.sep)

        rv = os.system(cmd)
                
        if not rv:
            logging.info('Finish downloading youtube video url {}'.format(video_url))
        else:
            logging.error('Unsuccessful downloading - youtube video url {}'.format(video_url))

        time.sleep(random.uniform(1.0, 1.5))

def download_ytvideo_multiple(saveto, MAX_THREADS=12) :
    content = json.load(open('WLASL_v0.3.json'))
    if not os.path.exists(saveto):
        os.mkdir(saveto)

    # Extract all YouTube video instances
    yt_instances = [
        inst for entry in content for inst in entry['instances']
        if 'youtube' in inst['url'] or 'youtu.be' in inst['url']
    ]

    # Use a thread pool to download videos concurrently
    with Pool(MAX_THREADS) as pool:
        pool.starmap(download_yt_videos, [(inst, saveto) for inst in yt_instances])

if __name__ == '__main__':
    check_youtube_dl_version()
    logging.info('Start downloading youtube videos.')
    
    thread_nonyt = threading.Thread(target=download_nonyt_videos_multiple, args=('raw_videos',))
    thread_yt = threading.Thread(target=download_ytvideo_multiple, args=('raw_videos',))

    thread_yt.start()
    thread_nonyt.start()

    thread_yt.join()
    thread_nonyt.join()

    print("FINISH..........................")
