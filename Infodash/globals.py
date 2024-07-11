# globals.py
import threading

# Initializing global lock variable
global_lock = threading.Lock()

# Initializing global macro variable for marking if macro_df is already stored in db
macro_df = False
