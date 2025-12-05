import os
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time
import numpy as np
import glob

class DZIHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to add more detailed logging"""
        print(f"[{self.address_string()}] {format % args}")

    def send_cors_headers(self):
        """Send CORS headers for all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')  # 24 hours

    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        print(f"Received request: {self.path}")
        
        if self.path == '/list_dzi':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            # Get all .dzi files from tiles directory
            dzi_files = []
            tiles_dir = 'tiles'
            if os.path.exists(tiles_dir):
                for file in os.listdir(tiles_dir):
                    if file.endswith('.dzi'):
                        name = os.path.splitext(file)[0]  # Remove .dzi extension
                        dzi_files.append({
                            'name': name,
                            'path': f'tiles/{file}'
                        })
            
            self.wfile.write(json.dumps(dzi_files).encode())
        else:
            # Serve static files as usual
            return SimpleHTTPRequestHandler.do_GET(self)

def open_browser():
    # Wait a short moment to ensure the server is running
    time.sleep(1.5)
    webbrowser.open('http://localhost:8000/viewer.html')

if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, DZIHandler)
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    print('Server running at http://localhost:8000')
    print('Opening viewer in your default browser...')
    httpd.serve_forever() 

