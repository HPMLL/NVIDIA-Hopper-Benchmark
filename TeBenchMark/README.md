# TeBenchMark

## Environment
* Use Docker
    * image: nvcr.io/nvidia/pytorch:23.08-py3
    * example
        
        ```bash
        docker run -it --gpus all -p 8900:8900 --name te --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /:/work nvcr.io/nvidia/pytorch:23.08-py3 /bin/bash
        
        docker start te

        docker exec -it te /bin/bash
        ```

## Linear
benchmark the performance of ```te.Linear```
```
python ./linear/linear.py
```

## TransformerLayer
benchmark the performance of ```te.TransformerLayer```
```
bash ./models/llama.sh
```

## LLM
benchmark the inference of LLM when using ```te.linear```

download the dataset
```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

```
bash llm/test.sh
```
