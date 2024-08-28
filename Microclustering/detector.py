import requests


class Detector:
    def __init__(self, endpoint: str = 'http://127.0.0.1:5000/compute'):
        self.endpoint = endpoint

    def set_endpoint(self, endpoint: str):
        self.endpoint = endpoint

    def evaluate(self, network: str, parameters: list) -> list:
        r = requests.post(url=self.endpoint + "/" + network,
                          json={"parameters": [str(v) for v in parameters]})

        json_response = r.json()

        #flat_list = [item for sublist in json_response for item in sublist]

        return json_response
