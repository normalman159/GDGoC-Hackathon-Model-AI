import os
import json
import shutil

def split_subset(content, subset_name, subset_size):
    subset = content[: subset_size]

    if not os.path.exists('data') :
        os.mkdir('data')
    
    if not os.path.exists(os.path.join('data', subset_name)) :
        os.mkdir(os.path.join('data', subset_name))

    for entry in subset :
        instances = entry['instances']
        for vid in instances :
            video_id = vid['video_id']

            src_video_path = os.path.join('videos', video_id + '.mp4')
            dst_video_path = os.path.join('data', subset_name, video_id + '.mp4')

            if not os.path.exists(src_video_path):
                print('{} does not exist!'.format(video_id))
                continue
            
            if os.path.exists(dst_video_path):
                print('{} exists.'.format(video_id))
                continue
        
            shutil.copyfile(src_video_path, dst_video_path)

    print("Finish................!!!")


def main() :
    content = json.load(open('WLASL_v0.3.json'))
    split_subset(content, 'WLASL100', 100)

if __name__ == "__main__" :
    main()