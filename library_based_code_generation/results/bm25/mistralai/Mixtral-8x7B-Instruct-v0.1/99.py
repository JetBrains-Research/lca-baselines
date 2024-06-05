 ```python
import os
import moviepy.editor as mpy
from moviepy.video.fx.all import crop, slow_motion, mask_and, fadeout, fadein
from moviepy.audio.fx.all import audio_fadeout, audio_fadein
import youtube_dl

# Check if video files exist, download if not
video_files = ['video.mp4', 'title_screen.mp4', 'credits_screen.mp4']
for video in video_files:
    if not os.path.isfile(video):
        ydl_opts = {'outtmpl': video}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download(['https://www.youtube.com/watch?v=dQw4w9WgXcQ'])
            except youtube_dl.DownloadError as e:
                print(f'Error downloading {video}: {e}')
                exit(1)

# Load audio, extract subclip, apply fade-in and fade-out
audio = mpy.AudioFileClip('audio.mp3')
subclip_audio = audio.subclip(10, 30)
subclip_audio = audio_fadein(subclip_audio, 2)
subclip_audio = audio_fadeout(subclip_audio, 2)

# Load video, extract subclip, crop
video_clip = mpy.VideoFileClip('video.mp4')
subclip_video = video_clip.subclip(10, 30)
crop_video = subclip_video.crop(x_center=video_clip.w/2, y_center=video_clip.h/2, width=video_clip.w/2, height=video_clip.h/2)

# Analyze audio and video to find tempo and looping segment
audio_tempo = find_video_period(subclip_audio)
loop_segment = crop_video.speedx(audio_tempo).loop(duration=subclip_audio.duration)

# Create mirrored video segment
mirrored_segment = mpy.ImageSequenceClip(loop_segment.iter_frames(), fps=loop_segment.fps)
mirrored_segment = mirrored_segment.fl_time(lambda t: loop_segment.duration - t)

# Combine original and mirrored video segments, apply fade-in and fade-out, add audio
final_clip = mpy.CompositeVideoClip([crop_video, mirrored_segment], size=crop_video.size)
final_clip = fadeout(fadein(final_clip, 2), 2)
final_clip = final_clip.set_audio(subclip_audio)

# Create title and credits screens
title_screen = mpy.VideoFileClip('title_screen.mp4')
credits_screen = mpy.ImageClip('credits_screen.png', duration=5).set_duration(5)

# Concatenate title screen, video, and credits screen
full_clip = mpy.concatenate_videoclips([title_screen, final_clip, credits_screen])

# Write final video to file
full_clip.write_videofile('output.mp4', fps=24, audio_bitrate='192k', bitrate='1200k')
```