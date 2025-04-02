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
        host="0.0.0.0",  # Change to localhost only
        port=10000,
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


import sqlite3
import json
import re

# class SqliteSaver:
#     def __init__(self, db_path="state_db/chat_history.db"):
#         print("🛠 Initializing SQLiteSaver...")
#         self.db_path = db_path  # Store DB path instead of keeping a shared connection
#         self._setup_database()
#         print("✅ SQLiteSaver initialized successfully!")

#     def _setup_database(self):
#         """Create database tables if they don't exist."""
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS state_versions (
#                     version INTEGER PRIMARY KEY AUTOINCREMENT,
#                     state TEXT NOT NULL,
#                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#                 )
#             """)
#             conn.commit()

#     def _get_connection(self):
#         """Return a new database connection."""
#         return sqlite3.connect(self.db_path, check_same_thread=False)

#     def get_next_version(self, *args, **kwargs):  # Accepts extra arguments safely
#         """Return the next version number safely."""
#         with self._get_connection() as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT MAX(version) FROM state_versions")
#             result = cursor.fetchone()[0]
#             return (result or 0) + 1

#     def _save_state(self, state):
#         """Handles serialization and database insertion."""
#         version = self.get_next_version()

#         def json_serializer(obj):
#             if isinstance(obj, dict):
#                 return {k: json_serializer(v) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [json_serializer(i) for i in obj]
#             elif hasattr(obj, '__dict__'):
#                 return json_serializer(obj.__dict__)
#             elif hasattr(obj, 'to_dict'):
#                 return json_serializer(obj.to_dict())
#             else:
#                 return str(obj)

#         try:
#             print(f"📝 Serializing state for version {version}...")
#             state_json = json.dumps(state, default=json_serializer)
#             print(f"✅ Successfully serialized state.")

#             with self._get_connection() as conn:
#                 cursor = conn.cursor()
#                 print("💾 Writing state to database...")
#                 cursor.execute(
#                     "INSERT INTO state_versions (version, state) VALUES (?, ?)",
#                     (version, state_json)
#                 )
#                 conn.commit()
#                 print(f"✅ State version {version} saved successfully!")
#                 return version
#         except Exception as e:
#             print(f"❗ SERIALIZATION ERROR: {e}")
#             return None

#     def put(self, state, *args, **kwargs):
#         """🔥 Save state as a JSON string in SQLite."""
#         return self._save_state(state)

#     def put_writes(self, state, *args, **kwargs):
#         """🔥 Same as put(), ensures compatibility."""
#         return self._save_state(state)

#     def get(self, version):
#         """Retrieve stored state safely."""
#         print(f"🔍 Retrieving state for version {version}...")
#         with self._get_connection() as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT state FROM state_versions WHERE version = ?", (version,))
#             result = cursor.fetchone()

#         if result:
#             print("✅ State found, converting back to dictionary...")
#             return json.loads(result[0])
#         else:
#             print("❗ No state found for this version!")
#             return None

#     def get_tuple(self, version):
#         """Retrieve stored state as a tuple (version, state)."""
#         print(f"🔍 Retrieving state as tuple for version: {version}")

#         if isinstance(version, dict):
#             def find_integer(d):
#                 """Recursively search for an integer inside a dictionary."""
#                 if isinstance(d, int):
#                     return d
#                 if isinstance(d, str):
#                     match = re.search(r'\d+', d)
#                     if match:
#                         return int(match.group())
#                 if isinstance(d, dict):
#                     for key, value in d.items():
#                         result = find_integer(value)
#                         if result is not None:
#                             return result
#                 return None

#             extracted_version = find_integer(version)
#             if extracted_version is not None:
#                 version = extracted_version
#                 print(f"✅ Using extracted version: {version}")
#             else:
#                 print("❌ ERROR: No valid integer found in dictionary.")
#                 return None

#         if not isinstance(version, int):
#             print(f"❌ ERROR: Expected integer for version, got {type(version)}: {version}")
#             return None

#         with self._get_connection() as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT version, state FROM state_versions WHERE version = ?", (version,))
#             result = cursor.fetchone()

#         if result:
#             print(f"✅ Tuple retrieved successfully: version {result[0]}")
#             return (result[0], json.loads(result[1]))
#         else:
#             print("❗ No state found for this version!")
#             return None

# # ✅ Use this checkpointer in LangGraph
# memory = SqliteSaver()
