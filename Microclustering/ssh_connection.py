# ssh_connection.py

from sshtunnel import SSHTunnelForwarder


def create_ssh_tunnel():
    ssh_user = 'theitger'  # 'bwulf'
    ssh_private_key = '/Users/theitger/.ssh/id_rsa'  # '/Users/bastianwulf/.ssh/id_rsa_uni'

    # SSH-Tunnel zum ersten Host (Jump Host) einrichten
    tunnel1 = SSHTunnelForwarder(
        ('sshjump1.uni-muenster.de', 22),
        ssh_username=ssh_user,
        ssh_pkey=ssh_private_key,
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

    tunnel1.start()
    tunnel2.start()

    return tunnel1, tunnel2


def stop_ssh_tunnels(tunnel1, tunnel2):
    tunnel1.stop()
    tunnel2.stop()
