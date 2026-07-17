import requests
 
with open("input.mp3", "rb") as f:
    files = {"file": ("input.mp3", f, "application/octet-stream")}
    data  = {"model": "Systran/faster-whisper-large-v3"}
    resp = requests.post(
        "http://192.168.136.89:5001/v1/audio/transcriptions",
        data=data,
        files=files,
    )

print("Status:", resp.status_code)
print("Body: ", resp.text)
print("HEADERS: ",dict(resp.headers))

print(requests.get("http://192.168.136.89:5001/v1/models").text)