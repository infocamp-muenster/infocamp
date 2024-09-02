import threading
from sshtunnel import SSHTunnelForwarder

class SSHTunnelThread(threading.Thread):
    def __init__(self, event):
        super().__init__()
        self.tunnel1 = None
        self.tunnel2 = None
        self.event = event

    def run(self):
        ssh_user = 'theitger'  # 'bwulf'
        ssh_private_key = '/Users/theitger/.ssh/id_rsa'  # '/Users/bastianwulf/.ssh/id_rsa_uni'
        self.tunnel1, self.tunnel2 = self.create_ssh_tunnel(ssh_user, ssh_private_key)
        self.tunnel1.start()
        self.tunnel2.start()

        # Set the event to signal that the SSH tunnel is established
        self.event.set()

    def stop(self):
        if self.tunnel1:
            self.tunnel1.stop()
        if self.tunnel2:
            self.tunnel2.stop()

    @staticmethod
    def create_ssh_tunnel(ssh_user, ssh_private_key):
        tunnel1 = SSHTunnelForwarder(
            ('sshjump1.uni-muenster.de', 22),
            ssh_username=ssh_user,
            ssh_pkey=ssh_private_key,
            remote_bind_address=('D-3160S21.uni-muenster.de', 2222),
            local_bind_address=('localhost', 2222)
        )

        tunnel2 = SSHTunnelForwarder(
            ('localhost', 2222),
            ssh_username='infoadmin',
            ssh_password='sKje0#4tZWw9h!',
            remote_bind_address=('localhost', 9200),
            local_bind_address=('localhost', 9200)
        )

        return tunnel1, tunnel2
