"""
Manager loras from url

@author: TechnikMax
@github: https://github.com/TechnikMax
"""
import hashlib
import os
import requests


def _hash_url(url):
    """Generates a hash value for a given URL."""
    return hashlib.md5(url.encode('utf-8')).hexdigest()


class LoraManager:
    """
    Manager loras from url
    """
    def __init__(self):
        self.cache_dir = "/models/loras/"

    def _download_lora(self, url):
        """
        Downloads a LoRa from a URL and saves it in the cache.
        """
        url_hash = _hash_url(url)
        filepath = os.path.join(self.cache_dir, f"{url_hash}.safetensors")
        file_name = f"{url_hash}.safetensors"

        if not os.path.exists(filepath):
            print(f"start download for: {url}")

            try:
                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Download successfully, saved as {file_name}")

            except Exception as e:
                raise Exception(f"error downloading {url}: {e}") from e

        else:
            print(f"LoRa already downloaded {url}")
        return file_name

    def check(self, urls):
        """Manages the specified LoRAs: downloads missing ones and returns their file names."""
        paths = []
        for url in urls:
            path = self._download_lora(url)
            paths.append(path)
        return paths
