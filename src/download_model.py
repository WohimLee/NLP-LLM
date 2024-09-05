#Model Download
from modelscope import snapshot_download
model_dir = snapshot_download('playmake/Baichuan2-7B-Chat', cache_dir='./')