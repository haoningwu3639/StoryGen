import os
import re
import cv2
import subprocess

from collections import defaultdict
import json

# video_path: the path to the videos.
# save_path: the path to save the extracted images
video_path = 'video'
save_path = 'image' 

# Get the total length of the video
def get_video_duration(video_fn):
    out = subprocess.check_output(["ffprobe", "-v", "quiet", "-show_format", "-print_format", "json", video_fn])
    ffprobe_data = json.loads(out)
    duration_seconds = float(ffprobe_data["format"]["duration"])
    return duration_seconds

# Get the type of the frames
def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

# Save the keyframes with type I
def save_i_keyframes(video_fn, v_id, s_path):
    frame_types = get_frame_types(video_fn) 
    i_frames = [x[0] for x in frame_types if x[1]=='I'] # x[0]: frame number, x[1]: frame type
    if i_frames:
        cap = cv2.VideoCapture(video_fn)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()

            # Calculate the timestamp of the keyframe
            milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
            seconds = milliseconds // 1000
            milliseconds = milliseconds % 1000
            minutes = 0
            hours = 0
            if seconds >= 60:
                minutes = seconds//60
                seconds = seconds % 60

            if minutes >= 60:
                hours = minutes//60
                minutes = minutes % 60
            # timestamp of the keyframe
            frame_time = str(int(hours))+'-'+str(int(minutes))+'-'+str(int(seconds))+'-'+str(int(milliseconds))
            # Save the image
            outname = str(v_id)+'_keyframe_'+str(frame_time)+'.jpg'
            save_name = s_path +'/' + outname
            cv2.imwrite(save_name, frame)
        cap.release()
    else:
        print ('No I-frames in '+video_fn)

def remove_tags(text):
    """
    Remove vtt markup tags
    """
    tags = [
        r'</c>',
        r'<c(\.color\w+)?>',
        r'<\d{2}:\d{2}:\d{2}\.\d{3}>',

    ]

    for pat in tags:
        text = re.sub(pat, '', text)

    # extract timestamp, only kep HH:MM
    text = re.sub(
        r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> .* align:start position:0%',
        r'\g<1>',
        #'',
        text
    )

    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
    return text

def remove_header(lines):
    """
    Remove vtt file header
    """
    pos = -1
    for mark in ('##', 'Language: en',):
        if mark in lines:
            pos = lines.index(mark)
    lines = lines[pos+1:]
    return lines

def merge_duplicates(lines):
    """
    Remove duplicated subtitles. Duplacates are always adjacent.
    """
    last_timestamp = ''
    last_cap = ''
    for line in lines:
        if line == "":
            continue
        if re.match('^\d{2}:\d{2}:\d{2}\.\d{3}$', line):
            if line != last_timestamp:
                yield line
                last_timestamp = line
        else:
            if line != last_cap:
                yield line
                last_cap = line
    
def merge_timestamp(lines):
    """
    Remove the end timestamp
    """
    for i in range(len(lines)):
        line = lines[i]
        if line == "":
            continue
        if i<len(lines)-1 and re.match('^\d{2}:\d{2}:\d{2}\.\d{3}$', line) and not re.match('^\d{2}:\d{2}:\d{2}\.\d{3}$', lines[i+1]) :
            yield line
        else:
            if not re.match('^\d{2}:\d{2}:\d{2}\.\d{3}$', line):
                yield line

if __name__ == '__main__':
    video_id = 0

    video_dictionary = defaultdict()
    
    for video_name in sorted(os.listdir(video_path)):
        v_path = video_path + "/" + video_name
        
        # basename is the name of the video, 0: name, 1: .mp4/.vtt
        if os.path.splitext(os.path.basename(v_path))[1] == '.vtt':
            print("Subtitle path : {}".format(v_path))            
            save_id = '%06d' % video_id 
            s_path = save_path+'/' +str(save_id)
            print("Save path : {}".format(s_path))
            os.makedirs(s_path, exist_ok=True)       
            
            s_path = s_path+ '/' + str(save_id) + ".txt"
            
            # Extract corresponding information from the vtt file
            with open(v_path) as f:
                text = f.read()
            lines = remove_tags(text).splitlines()
            lines = list(merge_duplicates(remove_header(lines)))
            lines = list(merge_timestamp(lines))
            j = 0
            # Write the information into a txt file
            with open(s_path, 'w') as f:
                for line in lines:
                    f.write(line)
                    if j % 2 == 0:
                        f.write("->")
                    else:
                        f.write("\n")
                    j += 1

        if os.path.splitext(os.path.basename(v_path))[1] == '.mp4':
            print("Video path : {}".format(v_path))            
            save_id = '%06d' % video_id 
            s_path = save_path+ '/' + str(save_id)
            print("Save path : {}".format(s_path))
            os.makedirs(s_path, exist_ok=True)
            
            # Save the keyframes to the target dir
            save_i_keyframes(v_path, save_id, s_path)
                                    
            video_id += 1

# python extract.py
