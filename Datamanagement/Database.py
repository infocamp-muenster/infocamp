import json
import requests
from elasticsearch import Elasticsearch, helpers
from sshtunnel import SSHTunnelForwarder


class Database:
    def __init__(self):
        self.es = Elasticsearch(['http://localhost:9200'])

    def upload(self, index, data):
        def chunk_document(doc, chunk_size):
            for i in range(0, len(doc), chunk_size):
                yield doc[i:i + chunk_size]

        def create_bulk_data(chunk):
            for entry in chunk:
                yield {
                    "_index": index,
                    "_source": entry
                }

        document = data

        chunk_size = 1000
        chunks = chunk_document(document, chunk_size)

        for idx, chunk in enumerate(chunks):
            bulk_data = create_bulk_data(chunk)
            success, failed = helpers.bulk(self.es, bulk_data, raise_on_error=False)

            if failed:
                print(f"Errors in chunk {idx + 1}:")
                for item in failed:
                    print(item)

    @staticmethod
    def create_ssh_tunnel(uniKennung, prvKey):
        # SSH-Tunnel zum ersten Host (Jump Host) einrichten
        tunnel1 = SSHTunnelForwarder(
            ('sshjump1.uni-muenster.de', 22),
            ssh_username=uniKennung,
            ssh_pkey=prvKey,
            remote_bind_address=('D-3160S21.uni-muenster.de', 2222),
            local_bind_address=('localhost', 2222)
        )

        # SSH-Tunnel zum zweiten Host (Zielserver) einrichten
        tunnel2 = SSHTunnelForwarder(
            ('localhost', 2222),
            ssh_username='infoadmin',
            ssh_password='sKje0#4tZWw9h!',
            remote_bind_address=('localhost', 9200),
            local_bind_address=('localhost', 9200)
        )

        return tunnel1, tunnel2

    @staticmethod
    def to_json(file):
        data = file.body if hasattr(file, 'body') else file
        # Convert the extracted data to JSON
        return json.dumps(data, indent=4)

    def search_get_all(self, index):
        # Initialize the scroll
        page = self.es.search(
            index=index,
            scroll='2m',
            size=1000,
            body={
                "query": {
                    "match_all": {}
                }
            }
        )

        sid = page['_scroll_id']
        scroll_size = page['hits']['total']['value']
        # Start collecting documents
        all_hits = page['hits']['hits']

        # Start scrolling
        while scroll_size > 0:
            page = self.es.scroll(scroll_id=sid, scroll='2m')
            # Update the scroll ID
            sid = page['_scroll_id']
            # Get the number of results that we returned in the last scroll
            scroll_size = len(page['hits']['hits'])
            # Add the results to the all_hits array
            all_hits.extend(page['hits']['hits'])

        return all_hits

    def upload_df(self, index, dataframe):
        try:
            # Split the DataFrame into chunks of 1000 entries each (adjust size as needed)
            chunk_size = 1000  # Number of entries per chunk
            chunks = self.chunk_dataframe(dataframe, chunk_size)

            # Index each chunk using the Bulk API
            for idx, chunk in enumerate(chunks):
                bulk_data = list(self.create_bulk_data_df(chunk, index))
                success, failed = helpers.bulk(self.es, bulk_data, raise_on_error=False)

                # Log detailed errors if there are any failures
                if failed:
                    print(f"Errors in chunk {idx + 1}:")
                    for item in failed:
                        print(item)

        except Exception as e:
            print(f"An Upload-error occurred: {e}")

    def update_df(self, index, dataframe, id_column):
        def prepare_bulk_data_df(df):
            for _, row in df.iterrows():
                doc = row.to_dict()
                if not self.is_record_existing(index, id_column, doc):
                    yield {
                        "_index": index,
                        "_source": doc
                    }

        try:
            # Split the DataFrame into chunks of 1000 entries each (adjust size as needed)
            chunk_size = 1000  # Number of entries per chunk
            chunks = self.chunk_dataframe(dataframe, chunk_size)

            # Index each chunk using the Bulk API
            for idx, chunk in enumerate(chunks):
                bulk_data = list(prepare_bulk_data_df(chunk))
                success, failed = helpers.bulk(self.es, bulk_data, raise_on_error=False)

                # Log detailed errors if there are any failures
                if failed:
                    print(f"Errors in chunk {idx + 1}:")
                    for item in failed:
                        print(item)

        except Exception as e:
            print(f"An Update-error occurred: {e}")

    def is_record_existing(self, index, id_column, record):
        query = {"query": {"term": {id_column: record[id_column]}}}
        response = requests.get(f"http://localhost:9200/{index}/_search", json=query).json()
        return response["hits"]["total"]["value"] > 0

    @staticmethod
    def chunk_dataframe(df, chunk_size):
        """Split the DataFrame into smaller chunks."""
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start:start + chunk_size]

    @staticmethod
    def create_bulk_data_df(df, index):
        """Prepare bulk data for indexing."""
        for _, row in df.iterrows():
            yield {
                "_index": index,
                "_source": row.to_dict()
            }

