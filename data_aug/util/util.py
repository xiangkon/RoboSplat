import subprocess


def save_rgb_images_to_video(images, output_filename, fps=30):
    height, width, layers = images[0].shape
    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', f'{width}x{height}',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               output_filename]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for image in images:
        process.stdin.write(image.tobytes())
    process.stdin.close()
    process.wait()
