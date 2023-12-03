import requests

resp = requests.post("http://localhost:5000/api/ai/predict",
                     json={"imageId":"0",
                            "imageName": "mdb012.pgm"
                            })


print(resp.json())
