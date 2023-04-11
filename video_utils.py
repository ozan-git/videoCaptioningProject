from torchvision.io.video import read_video, write_video
from pathlib import Path


def clip_video(video_path: Path, start_time: float, end_time: float, output_path: Path) -> None:
    """
    Clip a video from a given start time to end time.
    """
    # info[0] video fps info[1] audio fps
    clip_video, clip_audio, info = read_video(str(video_path), start_pts=start_time, end_pts=end_time, pts_unit="sec")
    video_fps = info['video_fps']
    audio_fps = info['audio_fps']
    write_video(filename=str(output_path), video_array=clip_video, fps=video_fps)

'''
videos = Path('./MSRVTT/TestVideo').glob('*')
if not Path('./MSRVTT/TestVideo_clips').exists():
    Path('./MSRVTT/TestVideo_clips').mkdir(parents=True, exist_ok=True)
for video in videos:
    clip_video(str(video), 0, 1, str(Path('./MSRVTT/TestVideo_clips/' + video.name + '_clip.mp4')))
    print(video.name)
'''
