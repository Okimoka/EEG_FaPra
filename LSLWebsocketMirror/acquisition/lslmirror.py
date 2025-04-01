import asyncio
import json
import random
import numpy as np
from websockets.server import serve

class LSLMirror:
    def __init__(self, discovery, port=8333):
        self.discovery = discovery
        self.web_stream_dict = {}
        self.port = port
        self.clients = set()
        self.running = False

    #async def _collect_data(self):
    #    """
    #    Collect data from streams and store in web_stream_dict.
    #    """
    #    while self.running:
    #        try:
    #            for uid, stream in self.discovery.streams_by_uid.items():
    #                info = self.discovery.info_by_uid[uid]
    #                key = info.name()
    #                
    #                for timestamp, samples in stream.buffer.process(timeout=0.0):
    #                    if timestamp and not np.any(np.isnan(samples)):
    #                        # Ensure samples are JSON serializable
    #                        if(0 in samples):
    #                            print("SENDING SACCADE!!! " + str(random.randint(0, 100)))
    #                        self.web_stream_dict[key] = {
    #                            "timeseries": samples,
    #                            "info": {
    #                                "nominal_srate": info.nominal_srate(),
    #                                "type": info.type(),
    #                                "channel_count": info.channel_count(),
    #                                "channel_format": info.channel_format(),
    #                                "source_id": info.source_id(),
    #                            },
    #                            "timestamp": timestamp,
    #                        }
    #            await asyncio.sleep(0.001)  
    #        except RuntimeError:
    #            
    #            pass

    async def _collect_data(self):
        """
        Collect data from streams and immediately broadcast each sample.
        """
        while self.running:
            try:
                for uid, stream in self.discovery.streams_by_uid.items():
                    info = self.discovery.info_by_uid[uid]
                    key = info.name()
                    # Access the latest samples efficiently
                    for timestamp, samples in stream.buffer.process(timeout=0.0):
                        if timestamp:
                            # Ensure samples are JSON serializable
                            data = {
                                key: {
                                    "timeseries": list(np.nan_to_num(np.array(samples))),  # Ensure JSON serializable
                                    "info": {
                                        "nominal_srate": info.nominal_srate(),
                                        "type": info.type(),
                                        "channel_count": info.channel_count(),
                                        "channel_format": info.channel_format(),
                                        "source_id": info.source_id(),
                                    },
                                    "timestamp": timestamp,
                                }
                            }
                            message = json.dumps(data)

                            if self.clients:
                                await asyncio.gather(
                                    *[client.send(message) for client in self.clients],
                                    return_exceptions=True
                                )
            except RuntimeError:
                # Dictionary changes size during iteration if devices are lost or new ones are found
                pass
            await asyncio.sleep(0.001) # Adjust as necessary
    
    #async def _broadcast(self):
    #    """
    #    Broadcast the latest data to all connected clients.
    #    """
    #    while self.running:
    #        if self.clients and self.web_stream_dict:
    #            message = json.dumps(self.web_stream_dict)
    #            await asyncio.gather(
    #                *[client.send(message) for client in self.clients],
    #                return_exceptions=True
    #            )
    #        await asyncio.sleep(0.01)  # Adjust as necessary

    async def _handler(self, websocket):
        """
        Handle new websocket connections.
        """
        print("Client connected:", websocket.remote_address)
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print("Client disconnected:", websocket.remote_address)
    
    async def start_server(self):
        """
        Start the WebSocket server and data processing tasks.
        """
        self.running = True
        data_task = asyncio.create_task(self._collect_data())
        #broadcast_task = asyncio.create_task(self._broadcast())

        async with serve(self._handler, "localhost", self.port) as server:
            print(f"WebSocket server started on ws://localhost:{self.port}")
            try:
                await asyncio.gather(
                    data_task,
                    #broadcast_task,
                    server.wait_closed(),
                )
            except asyncio.CancelledError:
                pass
            finally:
                # Clean up tasks on shutdown
                self.running = False
                data_task.cancel()
                #broadcast_task.cancel()
                await asyncio.gather(data_task, return_exceptions=True)

    def run(self):
        """
        Run the server.
        """
        try:
            asyncio.run(self.start_server())
        except KeyboardInterrupt:
            print("Server shutdown requested by user.")
