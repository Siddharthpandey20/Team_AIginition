import uvicorn
import os
import signal
import sys

def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)

def main():
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Use Render-provided port or default to 8000 for local testing
    port = int(os.getenv("PORT", 8000))  # Render sets PORT dynamically

    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()
