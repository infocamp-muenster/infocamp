import threading
import queue
from atproto import CAR, AtUri, FirehoseSubscribeReposClient, firehose_models, models, parse_subscribe_repos_message


class BlueskyFirehose:
    """
    Eine Klasse zur Verwaltung der Bluesky-Firehose-Verbindung und zur Verarbeitung von Posts
    basierend auf angegebenen Sprachfiltern.
    """

    def __init__(self, filters, workers_count=4, max_queue_size=500):
        """
        Initialisiert die BlueskyFirehose-Klasse.

        Args: filters (list): Eine Liste von Sprachfiltern, die auf die Posts angewendet werden.
        workers_count (int): Die Anzahl der Worker-Threads, die zur Verarbeitung der Nachrichten gestartet werden (Standard: 4).
        max_queue_size (int): Die maximale Größe der Warteschlange für eingehende Nachrichten (Standard: 500).
        """
        self.filters = filters
        self.workers_count = workers_count
        self.max_queue_size = max_queue_size
        self.client = FirehoseSubscribeReposClient(self.get_firehose_params())
        self.work_queue = queue.Queue(maxsize=self.max_queue_size)
        self.output_queue = queue.Queue()
        self.cancel_event = threading.Event()

    @staticmethod
    def get_firehose_params() -> models.ComAtprotoSyncSubscribeRepos.Params:
        """
        Gibt die Parameter für den Firehose-Client zurück.

        Returns:
            models.ComAtprotoSyncSubscribeRepos.Params: Die Parameter für den Firehose-Client.
        """
        return models.ComAtprotoSyncSubscribeRepos.Params(cursor=0)

    @staticmethod
    def _get_ops_by_type(commit: models.ComAtprotoSyncSubscribeRepos.Commit) -> dict:
        """
        Extrahiert und kategorisiert Operationen aus einer Commit-Nachricht.

        Args:
            commit (models.ComAtprotoSyncSubscribeRepos.Commit): Die zu verarbeitende Commit-Nachricht.

        Returns:
            dict: Ein Wörterbuch, das Operationen nach Typ (Posts, Reposts, Likes, Follows) und Aktion (erstellt, gelöscht) kategorisiert.
        """
        operation_by_type = {
            'posts': {'created': [], 'deleted': []},
            'reposts': {'created': [], 'deleted': []},
            'likes': {'created': [], 'deleted': []},
            'follows': {'created': [], 'deleted': []},
        }

        car = CAR.from_bytes(commit.blocks)

        for op in commit.ops:
            uri = AtUri.from_str(f'at://{commit.repo}/{op.path}')

            if op.action == 'create' and op.cid:
                create_info = {
                    'uri': str(uri),
                    'cid': str(op.cid),
                    'author': commit.repo,
                    'timestamp': commit.time,
                }
                record_raw_data = car.blocks.get(op.cid)
                if not record_raw_data:
                    continue

                record = models.get_or_create(record_raw_data, strict=False)

                if uri.collection == models.ids.AppBskyFeedLike and models.is_record_type(record,
                                                                                          models.ids.AppBskyFeedLike):
                    operation_by_type['likes']['created'].append({'record': record, **create_info})
                elif uri.collection == models.ids.AppBskyFeedPost and models.is_record_type(record,
                                                                                            models.ids.AppBskyFeedPost):
                    operation_by_type['posts']['created'].append({'record': record, **create_info})
                elif uri.collection == models.ids.AppBskyFeedRepost and models.is_record_type(record,
                                                                                              models.ids.AppBskyFeedRepost):
                    operation_by_type['reposts']['created'].append({'record': record, **create_info})
                elif uri.collection == models.ids.AppBskyGraphFollow and models.is_record_type(record,
                                                                                               models.ids.AppBskyGraphFollow):
                    operation_by_type['follows']['created'].append({'record': record, **create_info})

            elif op.action == 'delete':
                if uri.collection == models.ids.AppBskyFeedLike:
                    operation_by_type['likes']['deleted'].append({'uri': str(uri)})
                elif uri.collection == models.ids.AppBskyFeedPost:
                    operation_by_type['posts']['deleted'].append({'uri': str(uri)})
                elif uri.collection == models.ids.AppBskyFeedRepost:
                    operation_by_type['reposts']['deleted'].append({'uri': str(uri)})
                elif uri.collection == models.ids.AppBskyGraphFollow:
                    operation_by_type['follows']['deleted'].append({'uri': str(uri)})

        return operation_by_type

    def worker_main(self):
        """
        Hauptfunktion des Workers, um Nachrichten aus der Warteschlange zu verarbeiten und die Daten zu streamen.
        """
        while not self.cancel_event.is_set():
            message = self.work_queue.get()
            if message is None:
                break

            try:
                commit = parse_subscribe_repos_message(message)
                if not (commit and isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit) and commit.blocks):
                    continue

                ops = self._get_ops_by_type(commit)
                for post in ops.get('posts', {}).get('created', []):
                    post_msg = post['record'].text
                    post_langs = post['record'].langs
                    post_author = post['author']
                    post_timestamp = post['timestamp']
                    if any(filter_str in post_langs for filter_str in self.filters):
                        self.output_queue.put({
                            "Text": post_msg,
                            "Lang": post_langs,
                            "User": post_author,
                            "Timestamp": post_timestamp
                        })
            except Exception as e:
                print(f"Error processing message: {e}")

    def on_message_handler(self, message: firehose_models.MessageFrame):
        """
        Handler zum Hinzufügen von Nachrichten zur Arbeitswarteschlange.

        Args:
            message (firehose_models.MessageFrame): Der eingehende Nachrichtenrahmen.
        """
        try:
            self.work_queue.put(message)
        except Exception as e:
            print(f"Error in message handler: {e}")

    def stop_client_on_cancel(self):
        """
        Stoppt den Firehose-Client, wenn das Abbruchereignis gesetzt ist.
        """
        self.cancel_event.wait()
        self.client.stop()
        for _ in range(threading.active_count() - 1):
            self.work_queue.put(None)

    def run(self):
        """
        Startet die Datenverarbeitung mit den angegebenen Sprachfiltern.
        """
        for _ in range(self.workers_count):
            threading.Thread(target=self.worker_main, daemon=True).start()

        client_thread = threading.Thread(
            target=lambda: self.client.start(lambda message: self.on_message_handler(message)), daemon=True)
        client_thread.start()

        stop_thread = threading.Thread(target=self.stop_client_on_cancel, daemon=True)
        stop_thread.start()

    def stream(self):
        """
        Generator funktion zum Streamen von verarbeiteten Daten.

        Yields:
            dict: Ein Wörterbuch mit den Daten eines Posts.
        """
        while not self.cancel_event.is_set():
            item = self.output_queue.get()
            if item is None:
                break
            yield item

    def cancel(self):
        """
        Setzt das Abbruchereignis und leert die Ausgabewarteschlange.
        """
        self.cancel_event.set()
        for _ in range(threading.active_count() - 1):
            self.output_queue.put(None)

    def keyboard_listener(self):
        """
        Wartet auf eine Tastatureingabe, um den Stream abzubrechen.
        """
        input("Press any key to stop the stream...\n")
        self.cancel()
