
import requests

res = requests.get('https://www.youtube.com/@mitocw')
txt = res.text
status = res.status_code
print(txt, status)
