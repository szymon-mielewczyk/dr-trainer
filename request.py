import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"imageId":"0",
                            "imageName": "mdb012.pgm"
                            })

print(resp.json())
