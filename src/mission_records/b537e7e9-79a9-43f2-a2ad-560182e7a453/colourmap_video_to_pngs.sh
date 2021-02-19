#! To extract individual frames from the mp4
mkdir colourmap_video_frames
ffmpeg -i "colourmap_video.mp4" colourmap_video_frames/frame_%06d.png
