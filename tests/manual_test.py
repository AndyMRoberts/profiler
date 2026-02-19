from profiler import Profiler
import time
import os

# Create profiler: output base name, sampling frequency (Hz), title
output_dir = f'{os.path.dirname(os.path.abspath(__file__))}/runs'
p = Profiler(output_dir, frequency_hz=2.0, title="SLAM inference test")
print(f'Data will be saved to output directory: {output_dir}')

# Start recording (runs in background thread)
p.start()

# ... run your ML model, SLAM pipeline, etc. ...
time.sleep(5.0)
# Stop and write outputs; optionally pass num_frames for energy-per-frame stats
p.stop(num_frames=100)