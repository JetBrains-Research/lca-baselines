  import moviepy.editor as mpy
import subprocess

# Check if required files exist, download if not
required_files = ["audio.mp3", "video.mp4"]
for file in required_files:
    if not mpy.VideoFileClip(file).is_open():
        try:
            subprocess.run(["youtube-dl", file])
        except subprocess.CalledProcessError:
            print(f"Error: {file} not found and download failed")
            exit()

# Load audio and extract subclip
audio = mpy.AudioFileClip("audio.mp3")
audio_subclip = audio.subclip(0, 10)

# Apply fade-in and fade-out effects
audio_subclip = audio_subclip.volumex(0.5, 0.5)

# Analyze audio to find period
period = audio_subclip.get_duration()

# Load video and extract subclip
video = mpy.VideoFileClip("video.mp4")
video_subclip = video.subclip(0, 10)

# Crop video
video_subclip = video_subclip.crop(0, 0, 100, 100)

# Analyze video to find looping segment
looping_segment = video_subclip.detect_loops()[0]

# Extract looping segment and slow it down
looping_segment = looping_segment.set_fps(0.5)

# Make looping segment loop for duration of audio
looping_segment = looping_segment.set_duration(period)

# Create mirrored version of looping segment
mirrored_looping_segment = looping_segment.set_pos(("center", "center")).set_duration(period)

# Combine original and mirrored video segments side by side
combined_video = mpy.CompositeVideoClip([video_subclip, mirrored_looping_segment], size=(200, 100))

# Apply fade-in and fade-out effects
combined_video = combined_video.volumex(0.5, 0.5)

# Add audio to video
combined_video = combined_video.set_audio(audio_subclip)

# Create title screen with text overlay
title_screen = mpy.TextClip("Title Screen", font="Arial", fontsize=50, color="white")
title_screen = title_screen.set_pos(("center", "center"))

# Create credits screen with text on black background
credits_screen = mpy.TextClip("Credits", font="Arial", fontsize=50, color="white")
credits_screen = credits_screen.set_pos(("center", "center"))
credits_screen = credits_screen.set_bg(mpy.ColorClip(color=(0, 0, 0)))

# Concatenate title screen, video, and credits screen
final_video = mpy.concatenate_videoclips([title_screen, combined_video, credits_screen])

# Write final video to file
final_video.write_videofile("output.mp4", fps=30, audio_bitrate="192k", bitrate="1000k")