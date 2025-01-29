Setting up You-Quantified:

    In frontend folder:
    npm start 
    In backend folder:
    npm run dev
    In LSLWebsocketMirror-main/src folder:
    python main.py
    (only after all LSL streams are already running)


Looking at streams in matlab:

    addpath('/path/to/xdf-Matlab-master')
    streams = load_xdf('/path/to/file.xdf')

    figure;
    plot(streams{1}.time_stamps, streams{1}.time_series)
    streams{1}.info


fixations.xdf and slow_movement.xdf are sample streams containing EEG+Eye data
In slow_movement.xdf, the gaze moves slowly from corner to corner.
In fixations the gaze jumps around, fixating at each random spot for ~1s