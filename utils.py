import gdown
import os


_WEIGHTS_URL_CLIP = {
    'base-32': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
    'base-16': 'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
    'large-14': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'
}

_NAME_CLIP_FILE = {
    'base-32': 'ViT-B-32.pt',
    'base-16': 'ViT-B-16.pt',
    'large-14': 'ViT-L-14.pt'
}

def download_pretrained_clip(experiment):
    if experiment == "vl4str-large":
        url = _WEIGHTS_URL_CLIP['large-14']
        name = _NAME_CLIP_FILE['large-14']
    elif experiment == "vl4str-base32":
        url = _WEIGHTS_URL_CLIP['base-32']
        name = _NAME_CLIP_FILE['base-32']
    else:
        url = _WEIGHTS_URL_CLIP['base-16']
        name = _NAME_CLIP_FILE['base-16']
    
    if os.path.exists(f'./weight/vl4str/clip/{name}'):
        print("clip weight already exists")
    else:
        os.makedirs(f'./weight/vl4str/clip/', exist_ok=True)
        gdown.download(url, f'./weight/vl4str/clip/{name}', quiet=False)
        
