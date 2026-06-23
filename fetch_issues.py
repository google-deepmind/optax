import urllib.request
import json
with open("issues.txt", "w", encoding="utf-8") as f:
    try:
        req = urllib.request.Request('https://api.github.com/repos/google-deepmind/optax/issues?state=open&per_page=100', headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        issues = json.loads(response.read())
        for i in issues:
            if 'pull_request' not in i:
                labels = [l['name'] for l in i['labels']]
                f.write(f"#{i['number']}: {i['title']} (Labels: {labels})\n")
                f.write(f"URL: {i['html_url']}\n\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
