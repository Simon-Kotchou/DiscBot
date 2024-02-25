﻿# DiscBot

```
docker build -t waves-gpu .
## without volume 
docker run --gpus all \
     --env-file .env waves-gpu
## with volume (for model caching)
docker run -- gpus all \
     --env-file .env \
     -v my_volume:/root/.cache waves-gpu
```
