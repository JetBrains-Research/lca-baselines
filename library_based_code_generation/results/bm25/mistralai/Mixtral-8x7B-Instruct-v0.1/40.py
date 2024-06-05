 import nni
from nni.compression.pytorch.pruning import TaylorPruner, AGPPruner
from nni.compression.pytorch.quantization import QATQuantizer
from nni.compression.pytorch.distiller import DynamicLayerwiseDistiller
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils import PathA, PatternA
from nni.compression.pytorch.pytorch_utils import _model_and_evaluator
from nni.compression.pytorch.pytorch_utils import forward_and_backward
from nni.compression.pytorch.pytorch_utils import _replicate_and_instantiate
from nni.compression.pytorch.pytorch_utils import exploit_and_explore
from nni.compression.pytorch.pytorch_utils import set_stage_and_offload
from nni.compression.pytorch.pytorch_utils import TeacherModelBasedDistiller
from nni.compression.pytorch.pytorch_utils import test_simplify_and_random
from nni.compression.pytorch.pytorch_utils import _report_intermediates_and_final
from nni.compression.pytorch.pytorch_utils import _start_engine_and_strategy
from nni.compression.pytorch.pytorch_utils import _keep_trying_and_will_success
from nni.compression.pytorch.pytorch_utils import _get_layer_and_inputs_from_tuner
from nni.compression.pytorch.pytorch_utils import _build_model_for_step
from nni.compression.pytorch.pytorch_utils import path_of_module
from nni.compression.pytorch.pytorch_utils import patched_get_optimizer_cls_and_kwargs

import torch
import torchvision
import torchvision.transforms as transforms

def create_teacher_model(model):
return _replicate_and_instantiate(model)

def create_pruner(model):
pruner_config = {
"sparsity_spec": {"ops": ["Conv2d"], "target": 0.5},
"pruning_type": "unstructured",
"pruning_strategy": "level",
"training_steps": 100,
"total_runs": 30,
"init_scheme": "uniform",
"scheme_args": {"min_num_filters": 4},
"training_steps_per_run": 100,
"start_step": 0,
"mode": "iterative",
"patience": 5,
"factor": 0.1,
"granularity": "module",
"round_filters": True,
"operation_filter": PatternA([".*"]),
"name": "TaylorPruner+AGPPruner"
}
pruner = TaylorPruner(**pruner_config) | AGPPruner(**pruner_config)
return pruner

def create_quantizer(model):
quantizer_config = {
"quant_type": "QAT",
"quant_scheme": "symmetric",
"w_bits": 8,
"a_bits": 32,
"quant_grad": True,
"training_steps": 100,
"start_step": 100,
"operation_filter": PathA(["conv2d", "batchnorm2d"]),
"name": "QATQuantizer"
}
quantizer = QATQuantizer(**quantizer_config)
return quantizer

def create_distiller(model, teacher_model):
distiller_config = {
"distill_type": "layerwise",
"distill_method": "mse",
"training_steps": 60 * 100,
"start_step": 0,
"operation_filter": PathA(["conv2d"]),
"name": "DynamicLayerwiseDistiller"
}
distiller = DynamicLayerwiseDistiller(**distiller_config)
distiller.teacher_model = teacher_model
return distiller

def compress_model(model, pruner, quantizer, distiller):
engine, strategy = _start_engine_and_strategy()
tuner = exploit_and_explore(engine, strategy, model, pruner, quantizer, distiller)
_keep_trying_and_will_success(tuner)
return tuner.model

def speedup_model(model, pruner):
speedup = ModelSpeedup(model, pruner.get_masks())
return speedup.speedup()

def evaluate_accuracy(model, dataloader):
correct = 0
total = 0
with torch.no\_grad():
for data in dataloader:
images, labels = data
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
total += labels.size(0)
correct += (predicted == labels).sum().item()
return correct / total

# Fine-tune ResNet18 on Cifar10
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 10)
model, evaluator = _model_and_evaluator(model, torchvision.datasets.CIFAR10, transforms.Compose([transforms.ToTensor()]), 30)

# Fine-tuning
for epoch in range(30):
for data in evaluator.dataloader:
images, labels = data
outputs = model(images)
loss = torch.nn.functional.cross\_entropy(outputs, labels)
loss.backward()
evaluator.optimizer.step()
evaluator.optimizer.zero\_grad()

# Evaluate fine-tuned model
print("Fine-tuned model accuracy:", evaluate\_accuracy(model, evaluator.dataloader))

# Create teacher model
teacher\_model = create\_teacher\_model(model)

# Create pruner
pruner = create\_pruner(model)

# Create quantizer
quantizer = create\_quantizer(model)

# Create distiller
distiller = create\_distiller(model, teacher\_model)

# Compress model
compressed\_model = compress\_model(model, pruner, quantizer, distiller)

# Speedup model
speedup\_model = speedup\_model(compressed\_model, pruner)

# Evaluate compressed model
print("Compressed model accuracy:", evaluate\_accuracy(speedup\_model, evaluator.dataloader))

# Simulate quantization
quantizer.update\_calibration\_config(speedup\_model)

# Evaluate compressed and quantized model
print("Compressed and quantized model accuracy:", evaluate\_accuracy(quantizer.model, evaluator.dataloader))