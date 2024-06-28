import json
import csv
import os

# mappings.py
def map_data_to_json(data, timestamp_key, username_key, user_id_key, post_id_key, text_key):
    def convert_to_json(data):
        return json.dumps([
            {
                "timestamp": item.get(timestamp_key),
                "username": item.get(username_key),
                "user_id": item.get(user_id_key),
                "post_id": item.get(post_id_key),
                "text": item.get(text_key)
            }
            for item in data
        ], indent=4)

    return convert_to_json(data)