import argparse
import os

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Utility script for convering phoenix2014T frames into videos
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def mkdir_p(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(data_path):

    frames_path = "features/fullFrame-210x260px"
    sets = ["train", "dev", "test"]

    for s in sets:

        frames_set = os.path.join(data_path, frames_path, s)
        output_dir_videos = os.path.join("/home/nlp/dorink/project/bsl1k/data_phoenix", "videos", s)
        mkdir_p(output_dir_videos)

        for v in os.listdir(frames_set):

            frames_dir = os.path.join(frames_set, v)
            mp4_path = os.path.join(output_dir_videos, f"{v}.mp4")

            if not os.path.exists(mp4_path):
                # cmd_ffmpeg = (
                #     f'ffmpeg -y -threads 8 -r 25 -i '
                #     f'{os.path.join(frames_dir, "images%04d.png")}'
                #     f' -c:v h264 -pix_fmt yuv420p -crf 23 '
                #     f"{mp4_path}"
                #     ''
                # )
                # os.system(cmd_ffmpeg)
                os.system('ffmpeg -y -threads 8 -r 25 -i "{0}" -c:v h264 -pix_fmt yuv420p -crf 23 "{1}"'.format(
                    os.path.join(frames_dir, "images%04d.png"), mp4_path))

                # ffmpeg -y -threads 8 -r 25 -i /home/nlp/dorink/project/bsl1k/data/features/fullFrame-210x260px/train/23April_2010_Friday_tagesschau-3669/images%04d.png -c:v h264 -pix_fmt yuv420p -crf 23 /home/nlp/dorink/project/bsl1k/data_phoenix/videos/train/23April_2010_Friday_tagesschau-3669.mp4
                # ffmpeg -y -threads 8 -r 25 -i /home/nlp/dorink/project/bsl1k/data/features/fullFrame-210x260px/train/12December_2010_Sunday_tagesschau-5609/images%04d.png -c:v png -pix_fmt yuv420p -crf 23 /home/nlp/dorink/project/bsl1k/data_phoenix/videos/train/12December_2010_Sunday_tagesschau-5609.mp4
                # 'ffmpeg -y -threads 8 -r 25 -i "{0}" -c:v h264 -pix_fmt yuv420p -crf 23 "{1}"'.format(os.path.join(frames_dir, "images%04d.png"), mp4_path))
                # ./ffmpeg -i Image%03d.png -c:v mpeg4 -qscale:v 5 -r 20 -b:v 9600k movie.mp4
                # ffmpeg -i Image%03d.png -pix_fmt yuv420p -c:v libx264 -crf 23 movie.mp4


if __name__ == "__main__":

    description = ("Helper script for combining the original Phoenix frames into mp4 videos.")

    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--data_path",
        type=str,
        default="/home/nlp/dorink/project/bsl1k/data",  # data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T",
        help="Path to Phoenix data.",
    )

    main(**vars(p.parse_args()))
