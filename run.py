import uvicorn
import webbrowser
import threading
import time
import signal
import sys

def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)

def open_browser():
    """Open browser after server starts"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:8000')

def main():
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Configure and run server
    config = uvicorn.Config(
        "main:app",
        host="127.0.0.1",  # Change to localhost only
        port=8000,
        reload=True,
        access_log=True,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        print("Cleanup complete")

if __name__ == "__main__":
    main()

