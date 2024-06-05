```python
import os
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.video.fx.loop import loop
from moviepy.video.fx.resize import resize
from moviepy.audio.fx.all import audio_fadein, audio_fadeout

# 1. Check and download video files
video_filenames = ["video.mp4"]
audio_filename = "audio.mp3"

for filename in video_filenames + [audio_filename]:
    if not os.path.exists(filename):
        try:
            os.system(f"youtube-dl {filename} -o {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            exit(1)

# 2. Load and process audio
audio = AudioFileClip(audio_filename).subclip(10, 70)
audio = audio_fadein(audio, 3).audio_fadeout(3)
audio_period = audio.duration  # Simplified approach to find period

# 3. Load and process video
video = VideoFileClip(video_filenames[0]).subclip(50, 120).crop(x1=100, y1=100, x2=400, y2=400)
video_period = video.duration  # Simplified approach to find a looping segment

# 4. Extract looping segment, slow down, and loop
loop_video = loop(video, duration=audio.duration)
mirrored_video = loop_video.fx(vfx.mirror_x)

# 5. Combine original and mirrored video, add audio
combined_clip = CompositeVideoClip([loop_video, mirrored_video.set_position(("right", 0))])
combined_clip = combined_clip.set_audio(audio)
combined_clip = combined_clip.fx(vfx.fadein, 3).fx(vfx.fadeout, 3)

# 6. Create title and credits screens
title = TextClip("Title", fontsize=70, color='white', size=combined_clip.size).set_duration(5).set_position("center").set_fps(24)
credits = TextClip("Credits", fontsize=70, color='white', size=combined_clip.size).set_duration(5).set_position("center").set_fps(24).on_color(color=(0,0,0), col_opacity=1)

# 7. Concatenate title, video, and credits into final video
final_clip = concatenate_videoclips([title, combined_clip, credits])

# 8. Write final video to file
final_clip.write_videofile("final_video.mp4", fps=24, audio_bitrate="192k", bitrate="2000k")
```