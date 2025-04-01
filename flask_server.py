from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import threading
import time


"""
Opens a Flask server on port 5000 to allow for communication with YouQuantified
"""

class FlaskServer:
    def __init__(self, post_event_handler, event_function, init_function):
        self.app = Flask(__name__, static_folder='static')
        # Need to set CORS policy so we don't get blocked
        CORS(self.app)
        # Setup Websocket
        # We need to use subprocesses, since socketio needs to run on a main thread and needs to be non-blocking
        # Handling of the subprocess has been offset to main.py, to keep this class closer to the init/start.. paradigm
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        # Unused right now, for POST requests
        self.data_ready = threading.Event()
        self.post_event_handler = post_event_handler
        self.event_function = event_function
        self.init_function = init_function
        
        # Unused, could allow for serving images to YouQuantified
        self.app.add_url_rule('/event', 'event', self.event_function, methods=['POST'])
        self.app.add_url_rule('/images/<path:path>', 'serve_image', self.serve_image)

        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')

    def serve_image(self, path):
        return send_from_directory('static', path)

    def initialize(self):
        self.init_function()

    def start(self):
        #Listen for POST events
        threading.Thread(target=self._event_loop, daemon=True).start()
        #Handle queue for Subprocess
        threading.Thread(target=self.push_data, daemon=True).start()
        self.socketio.run(self.app, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)

    def _event_loop(self):
        try:
            while True:
                time.sleep(1)
                if self.data_ready.is_set():
                    self.event_function()
                    self.data_ready.clear()
        except KeyboardInterrupt:
            print("Shutting down Flask Server")

    def set_data_ready(self):
        self.data_ready.set()

    def clear_data_ready(self):
        self.data_ready.clear()

    def set_queue(self, queue):
        self.queue = queue

    def push_data(self):
        while True:
            if self.queue and not self.queue.empty():
                data = self.queue.get()
                self.socketio.emit('data', data)
            time.sleep(0.1)
