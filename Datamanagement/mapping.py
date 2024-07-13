import pandas as pd

def map_data_to_dataframe(data, timestamp_key, username_key, user_id_key, post_id_key, text_key):
    mapped_data = [
        {
            "created_at": item.get(timestamp_key),
            "user_screen_name": item.get(username_key),
            "user_id_str": item.get(user_id_key),
            "id_str": item.get(post_id_key),
            "text": item.get(text_key)
        }
        for item in data
    ]
    return pd.DataFrame(mapped_data)