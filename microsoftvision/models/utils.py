import torch
import os
import tempfile
from urllib.request import urlopen, Request
import shutil
from azure.identity import DeviceCodeCredential
from azure.storage.blob import BlobClient
from tqdm import tqdm

def download_model_from_blob(url, dst, progress=True):
    blob_client = BlobClient.from_blob_url(url)
    model_size = int(blob_client.get_blob_properties()['size'])
    try:
        print(f"Model size: {model_size//(1024*1024)} MB")
        with open(dst, "wb") as my_blob:
            segment_size = 4 * 1024 * 1024 # 1MB chunk
            offset = 0
            for i in tqdm(range((model_size // segment_size) + 1), unit='MB'):
                if offset >= model_size:
                    break
                download_stream = blob_client.download_blob(offset=offset, length=segment_size)
                my_blob.write(download_stream.readall())
                offset += segment_size

    except:
        os.remove(dst)
        print("Downloading error")
        raise

    print(f"Model saved to {dst}")


def load_state_dict_from_url(model_path, map_location=None):
    filename = os.path.basename(model_path)

    # This checks if model exists in current folder
    if not os.path.exists(filename):
        print("Downloading model.")
        download_model_from_blob(model_path, filename)
    else:
        print("Model already downloaded.")

    return torch.load(filename, map_location=map_location)['state_dict']

def load_state_dict(model, pretrained_weights):
    weights = model.state_dict()

    # Remove non-exist keys
    for key in pretrained_weights.keys() - weights.keys():
        print("Delete unused model state key: %s" % key)
        del pretrained_weights[key]

    # Remove keys that size does not match
    for key, pretrained_weight in list(pretrained_weights.items()):
        weight = weights[key]
        if pretrained_weight.shape != weight.shape:
            print("Delete model state key with unmatched shape: %s" % key)
            del pretrained_weights[key]

    # Copy everything that pretrained_weights miss
    for key in weights.keys() - pretrained_weights.keys():
        print("Missing model state key: %s" % key)
        pretrained_weights[key] = weights[key]

    # Load the weights to model
    model.load_state_dict(pretrained_weights)
