from main import SqliteSaver

# ✅ Initialize SQLiteSaver
memory = SqliteSaver()

# ✅ Quick Check 1: Test get_next_version()
print("\n🔍 TESTING get_next_version()...")
try:
    version = memory.get_next_version("abc123")  # ✅ Pass a thread_id
    print(f"✅ SUCCESS: Next version = {version}")
except Exception as e:
    print(f"❌ ERROR in get_next_version(): {e}")

# ✅ Quick Check 2: Test get_tuple() with an invalid version type
print("\n🔍 TESTING get_tuple() with dictionary input...")
try:
    test_input = {"tags": [], "metadata": {"thread_id": "abc123"}}
    result = memory.get_tuple("abc123")  # ✅ Pass thread_id instead of dict
    print(f"✅ SUCCESS: get_tuple() handled thread_id correctly. Output: {result}")
except Exception as e:
    print(f"❌ ERROR in get_tuple(): {e}")

# ✅ Quick Check 3: Insert dummy data and retrieve it
print("\n🔍 TESTING database insert and retrieve...")
try:
    test_state = {"example": "test_data"}
    memory.put("abc123", test_state)  # ✅ Ensure thread_id is passed
    retrieved = memory.get("abc123")
    print(f"✅ SUCCESS: Retrieved state = {retrieved}")
except Exception as e:
    print(f"❌ ERROR in database operations: {e}")

# ✅ Insert test data and retrieve
print("\n🔍 TESTING manual data insertion and retrieval...")
try:
    memory.put("test_thread", {"message": "Hello, this is a test"})
    result = memory.get_tuple("test_thread")
    print(f"✅ SUCCESS: Retrieved state = {result}")
except Exception as e:
    print(f"❌ ERROR in test data retrieval: {e}")

print("\n🎯 QUICK TESTS COMPLETED!")
