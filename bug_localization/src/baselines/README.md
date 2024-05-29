# Baselines

```shell
python +data_src=hf data_src.hub_name=tiginamaria/bug-localization +backbone=openai +backbone/prompt=detailed backbone.model_name=gpt-3.5-turbo-16k ++backbone.parameters.temperature=0.8 ++backbone.parameters.seed=2687987020 logger.name=gpt_3.5_16k-detailed
```
