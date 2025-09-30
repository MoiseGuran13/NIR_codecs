import subprocess
import cv2
import os
import argparse

def encode(video_path, aedat_name="aedat", output_path="outputs", tresh=0.15, sigma_tresh=0.03):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    subprocess.run(f"python v2e.py -i {video_path} --overwrite --dvs_aedat4 {aedat_name}.aedat --disable_slomo --dvs_params clean --auto_timestamp_resolution=True --dvs_exposure duration 0.005 --output_folder={output_path} --overwrite --pos_thres={tresh} --neg_thres={tresh} --sigma_thres={sigma_tresh} --output_width={width} --output_height={height}".split(" "))
    os.remove(os.path.join(output_path, "dvs-video-frame_times.txt"))
    os.remove(os.path.join(output_path, "dvs-video.avi"))
    os.remove(os.path.join(output_path, "v2e-args.txt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='event2im evaluation script')
    parser.add_argument('-i', '--input', type=str, help='input video path')
    parser.add_argument('-o', '--output', type=str, help='output video name', default="nimic")
    parser.add_argument('-op', '--out_path', type=str, help='output path', default="")
    parser.add_argument('-t', '--tresh', type=float, help='positive and negative thresholds', default=0.15)
    parser.add_argument('-s', '--sigma', type=float, help='sigma treshold', default=0.03)
    args = parser.parse_args()
    if args.output == "nimic":
        output = args.input.split(".")[0]
    else:
        output = args.output

    encode(args.input, output, args.out_path, args.tresh, args.sigma)

