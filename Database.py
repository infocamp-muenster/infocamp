import json
from elasticsearch import Elasticsearch, helpers
from sshtunnel import SSHTunnelForwarder


class Database:
    @staticmethod
    def __init__(self):
        self.es = Elasticsearch(['http://localhost:9200'])

    @staticmethod
    def upload(self, index, file):
        es = Elasticsearch(['http://localhost:9200'])

        def chunk_document(doc, chunkSize):
            """Split the document into smaller chunks."""
            for i in range(0, len(doc), chunkSize):
                yield doc[i:i + chunkSize]

        def create_bulk_data(chunk):
            """Prepare bulk data for indexing."""
            for entry in chunk:
                yield {
                    "_index": index,
                    "_source": entry
                }

        # Read the JSON document from the file with UTF-8 encoding
        with open(file, 'r', encoding='utf-8') as file:
            document = json.load(file)

        # Split the document into chunks of 1000 entries each (adjust size as needed)
        chunk_size = 1000  # Number of entries per chunk
        chunks = chunk_document(document, chunk_size)

        # Index each chunk using the Bulk API
        for idx, chunk in enumerate(chunks):
            bulk_data = create_bulk_data(chunk)
            success, failed = helpers.bulk(es, bulk_data, raise_on_error=False)

            # Log detailed errors if there are any failures
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
    def toJSON(self, file):
        data = file.body if hasattr(file, 'body') else file
        # Convert the extracted data to JSON
        toJSON = json.dumps(data, indent=4)
        return toJSON

    @staticmethod
    def searchGetAll(self, index):
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
