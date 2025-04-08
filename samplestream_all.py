import numpy as np
import pyxdf
import time
from pylsl import StreamInfo, StreamOutlet
import threading

"""
Script mostly written by ChatGPT

This replays the supplied xdf stream in a constant loop into LSL Outlets.
It stays true to the original speed of the recording.
Very useful so we don't have to wear the cap + glasses for testing

"""


# Load the .xdf file
file_path = 'fixations.xdf'
streams, fileheader = pyxdf.load_xdf(file_path)

# Function to create an LSL outlet for each stream
def create_outlet(stream):
    # Create meta-info for the outlet (name, type, channel count, sampling rate, channel format, source_id)
    info = StreamInfo(name=stream['info']['name'][0],
                      type=stream['info']['type'][0],
                      channel_count=int(stream['info']['channel_count'][0]),
                      nominal_srate=float(stream['info']['nominal_srate'][0]),
                      channel_format='float32',
                      source_id=stream['info']['source_id'][0])
    # Create the outlet
    outlet = StreamOutlet(info)
    return outlet

# Create a dictionary to hold the outlets
outlets = {stream['info']['name'][0]: create_outlet(stream) for stream in streams}

# Function to stream data with real-time pacing for a single stream
def stream_single_stream(stream):
    outlet = outlets[stream['info']['name'][0]]
    data = stream['time_series']
    timestamps = stream['time_stamps']
    if timestamps.size > 0:
        while True:  # Loop indefinitely
            initial_time = timestamps[0]
            start_time = time.time()

            # Iterate over all samples in the stream
            for i in range(len(data)):
                current_real_time = time.time()
                elapsed_real_time = current_real_time - start_time
                elapsed_recorded_time = timestamps[i] - initial_time

                # Delay to match the original recording timing
                if elapsed_recorded_time > elapsed_real_time:
                    time.sleep(elapsed_recorded_time - elapsed_real_time)

                # Push the sample with its timestamp
                #print(f"Pushing sample {i} from stream {stream['info']['name'][0]}")
                outlet.push_sample(data[i], time.time())#timestamps[i])

            print(f"Stream {stream['info']['name'][0]} finished. Restarting...")
    else:
        print(f"Stream {stream['info']['name'][0]} has no timestamps, skipping.")

# Function to start streaming all streams in separate threads
def stream_data():
    print("Starting streaming...")
    threads = []

    # Create and start a thread for each stream
    for stream in streams:
        thread = threading.Thread(target=stream_single_stream, args=(stream,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish (they won't, since they loop indefinitely)
    for thread in threads:
        thread.join()

# Start streaming
if __name__ == '__main__':
    try:
        stream_data()
    except KeyboardInterrupt:
        print("Streaming stopped.")