import os
import requests

categories = {
    "bread": [
        "https://images.unsplash.com/photo-1608198093002-ad4e005484ec",
        "https://images.unsplash.com/photo-1589927986089-35812388d1f4"
    ],
    "milk": [
        "https://images.unsplash.com/photo-1563636619-e9143da7973b",
        "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b"
    ],
    "butter": [
        "https://images.unsplash.com/photo-1589985270958-bd4d53c2e6b7",
        "https://images.unsplash.com/photo-1604908176997-125f25cc6f3d"
    ]
}

base_path = "data/raw/images"

for category, urls in categories.items():
    folder = f"{base_path}/{category}"
    os.makedirs(folder, exist_ok=True)

    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"{folder}/{i}.jpg", "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {category} image {i}")
        except:
            print(f"Erreur téléchargement {url}")