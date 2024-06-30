import hashlib
import os
import requests
import tarfile

def _hash_url(url):
    """Generates a hash value for a given URL."""
    return hashlib.md5(url.encode('utf-8')).hexdigest()

class LoraManager:
    """
    Manager loras from url
    """
    def __init__(self):
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../',
            'repositories/Fooocus/models/loras')

    def _download_lora(self, url):
        """
        Downloads a LoRa from a URL, saves it in the cache, and if it's a .tar file, extracts it and returns the .safetensors file.
        """
        url_hash = _hash_url(url)
        file_ext = url.split('.')[-1]
        filepath = os.path.join(self.cache_dir, f"{url_hash}.{file_ext}")

        if not os.path.exists(filepath):
            print(f"Start download for: {url}")

            try:
                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                if file_ext == "tar":
                    print("Extracting the tar file...")
                    with tarfile.open(filepath, 'r:*') as tar:
                        tar.extractall(path=self.cache_dir)
                    print("Extraction completed.")
                    return self._find_safetensors_file(self.cache_dir)

                print(f"Download successfully, saved as {filepath}")
            except Exception as e:
                raise Exception(f"Error downloading {url}: {e}") from e

        else:
            print(f"LoRa already downloaded {url}")

        return filepath

    def _find_safetensors_file(self, directory):
        """
        Finds the first .safetensors file in the specified directory.
        """
        print("Searching for .safetensors file.")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.safetensors'):
                    return os.path.join(root, file)
        raise FileNotFoundError("No .safetensors file found in the extracted files.")

    def check(self, urls):
        """Manages the specified LoRAs: downloads missing ones and returns their file names."""
        paths = []
        for url in urls:
            path = self._download_lora(url)
            paths.append(path)
        return paths
