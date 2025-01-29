


live_dot.py
uses a stripped-down version pygame_visualization that only draws the gaze
mostly for testing for now. visualizes the fixations by color


fixations_stream.py creates a binary LSL outlet for fixations

main.py
- creates an LSL outlet for surface gaze and blinks
- opens a webserver to communicate with yq and serve images
- all main calculations (unfoldjl...)