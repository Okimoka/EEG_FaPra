from pylsl import StreamInlet, resolve_byprop, resolve_streams

class LSLStream:
    def __init__(self, name, track_history_seconds=0):

        self.inlet = None
        self.history = []
        self.name = name
        self.track_history_seconds = track_history_seconds
        self.connected = False

        if(name == None):
            self.connected = True
            return

        if(name in [stream.name() for stream in resolve_streams()]):
            self.inlet = StreamInlet(resolve_byprop('name', name)[0])
            print("Connected to stream: " + name)
            self.connected = True
        else:
            print("Stream " + name + " not found.")
        
    def clear_history_to_index(self, index):
        self.history = self.history[index:]

    
    def pull_sample(self):

        sample, timestamp = self.inlet.pull_sample(timeout=0.5)
        # bad hard coded solution
        # for the Neon Gaze stream we are really only interested in these three numbers for now
        if(self.name == "ccs-neon-001_Neon Gaze"):
            sample = sample[6:9] #x,y,z optical axis of the left eye

        if sample:
            if self.track_history_seconds > 0:
                # Detect time loop / jump
                if self.history and (self.history[-1][0]-timestamp)>0.02: #allow for some jitter
                    print(f"Timestamp jump detected in {self.name}. Clearing history." + str(timestamp-self.history[-1][0]))
                    self.history.clear()

                # Check for non monotonically increasing timestamps
                if self.history and (timestamp - self.history[-1][0]) < 0:
                    if(timestamp - self.history[-1][0]) < -0.01: #some negative numbers are unavoidable due to sampling rate
                        print("Skipping sample with " + str(timestamp - self.history[-1][0]) + " seconds delay")
                else:
                    self.history.append((timestamp, sample))

                # Remove samples older than history window
                while self.history and (timestamp - self.history[0][0]) > self.track_history_seconds:
                    self.history.pop(0)
                
                return timestamp, sample
        else:
            #print("Stream "+ self.name +" is not sending data!")
            return (0,[0])
    
    #Pulling chunks should be more efficient
    def pull_chunk(self):
        try:
            chunk, timestamps = self.inlet.pull_chunk()

            if(self.name == "ccs-neon-001_Neon Gaze"):
                chunk = [sample[6:9] for sample in chunk]
            
            if chunk:
                if self.track_history_seconds > 0:
                    self.history.extend(zip(timestamps, chunk))
                    # Remove samples that are older than the specified track_history_seconds
                    while self.history and (timestamps[-1] - self.history[0][0]) > self.track_history_seconds:
                        self.history.pop(0)
                return timestamps, chunk
            else:
                #print("Stream "+ self.name +" is not sending data!")
                return ([],[[]])
        except Exception as e:
            #print(e) Todo
            print("Warning: Pulling chunk from "+ self.name +" failed")
            return ([],[[]])

