# Python Mirror From LSL Into a WebSocket
This is a small Python script that I wrote to mirror an [LSL Stream](https://github.com/sccn/labstreaminglayer) into a WebSocket for the [You: Quantified project](https://github.com/esromerog/QuantifiedSelf). This way, you can interface with local LSL physiological data streams (it was tested using EEG and Audio Streams) into the Quantified Self Web App. 


#### Running the script
1. First, you make sure you have the correct packages installed. The project is lightweight, so you could install them separately or using `python -m pip install -r requirements.txt`.
2. To run the script, run the src/main.py using the `python3 src/main.py` command or run the file manually. Make sure you are within an environment that has the packages installed. For more information, you can look at [Python's Packaging tutorials](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)


#### Using it within the App
LSL functionality in the Quantified Self app is currently hidden to undergo more user testing. To enable it, hold the `alt` key while adding a new device.

#### Additional Information
This script is based on the acquisition and processing from [Hybrid Harmonny](https://github.com/RhythmsOfRelating/HybridHarmony), which should allow more complicated processing functionality in the future.

