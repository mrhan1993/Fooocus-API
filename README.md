# Fooocus-API (Under development)
FastAPI powered API for [Fooocus](https://github.com/lllyasviel/Fooocus)

### Install dependencies.
Need python version >= 3.10
```
pip install -r requirements.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118 xformers
```
You may change the part "cu118" of extra-index-url to your local installed cuda driver version.

### Sync dependent and download models (Optional)
```
python main.py --sync-repo only
```
After run successful, you can see the terminal print where to put the model files for Fooocus.

Then you can put the model files to target directories manually, or let it auto downloads when start app.

### Start app
```
python main.py
```
or
```
python main.py --host 0.0.0.0 --port 3333
```

For more pragram arguments, see
```
python main.py -h
```