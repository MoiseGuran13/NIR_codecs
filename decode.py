import subprocess
import cv2
import os
import aedat
import numpy as np
import argparse

MAGIC_PATH_OUT = "outputs\\std_all\\TEMP\\seq0"
MAGIC_PATH_DATA = "data\\TEMP\\seq0"

def events2voxels(file_path, video_path):
    for flag in ["ps", "xys", "tss"]:
        print(flag)
        decoder = aedat.Decoder(file_path)

        ps = []
        tss = []
        xys = []

        for packet in decoder:
            # print(packet)
            if 'events' in packet:
                # print(packet['events'].dtype)
                for event in packet['events']:
                    if flag == "ps":
                        ps.append(1 if event[3]==True else 0)
                    elif flag == "tss":
                        tss.append(event[0].astype(np.float64) / 1e6)
                    else:
                        xys.append([event[1], event[2]])

        if flag == "ps":
            np.save(os.path.join(MAGIC_PATH_DATA, "events_p.npy"), ps)
        elif flag == "tss":
            np.save(os.path.join(MAGIC_PATH_DATA, "events_ts.npy"), tss)
        else:
            np.save(os.path.join(MAGIC_PATH_DATA, "events_xy.npy"), xys)


    event_ts = tss

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video loaded: {frame_count} frames at {fps:.2f} FPS ({duration:.2f} seconds)")

    images_ts = np.arange(0, frame_count) / fps  

    images = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        images.append(frame)
    cap.release()

    images = np.stack(images)  
    image_event_indices = np.searchsorted(event_ts, images_ts, side='left')  
    images_ts = images_ts.reshape(-1, 1)
    image_event_indices = image_event_indices.reshape(-1, 1)

    np.save(os.path.join(MAGIC_PATH_DATA, "images.npy"), images)
    np.save(os.path.join(MAGIC_PATH_DATA, "images_ts.npy"), images_ts)
    np.save(os.path.join(MAGIC_PATH_DATA, "image_event_indices.npy"), image_event_indices)


    with open(os.path.join(MAGIC_PATH_DATA, "metadata.json"), "w") as jsonfile:
        jsonfile.write("{\"sensor_resolution\": [" + str(height) + ", " + str(width) + "]}")


def move_video(output_name:str, output_path:str):
    for file in os.listdir(MAGIC_PATH_OUT):
        if file.endswith(".mp4"):
            os.rename(os.path.join(MAGIC_PATH_OUT, file), os.path.join(output_path, output_name + ".mp4"))
    

def remove_dir(path:str):
    for file in os.listdir(path): 
        new_path = os.path.join(path, file)
        if os.path.isdir(new_path):
            remove_dir(new_path)
        else:
            os.remove(new_path)
    os.rmdir(path)
        


def decode(events_path:str, video_path:str, output_path:str, output_name:str, model:str):
    os.mkdir(MAGIC_PATH_OUT)
    os.mkdir(MAGIC_PATH_DATA)

    events2voxels(events_path, video_path)
    subprocess.run(f"python eval.py -qm mse -m {model} -c std_all -d TEMP".split(" "))
    move_video(output_name, output_path)
    remove_dir(MAGIC_PATH_DATA)
    remove_dir(MAGIC_PATH_OUT)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='event2im evaluation script')
    parser.add_argument('-i', '--input', type=str, help='input aedat4 file')
    parser.add_argument('-or', '--original_video', type=str, help='original mp4 video')
    parser.add_argument('-o', '--output', type=str, help='output video name', default="nimic")
    parser.add_argument('-op', '--out_path', type=str, help='output path', default="")
    parser.add_argument('-m', '--model', type=str, help='model used for decoding', default="E2VID")
    args = parser.parse_args()
    if args.output == "nimic":
        output = args.input.split(".")[0]
    else:
        output = args.output
    decode(args.input, args.original_video, args.out_path, output, args.model)
    
    
    