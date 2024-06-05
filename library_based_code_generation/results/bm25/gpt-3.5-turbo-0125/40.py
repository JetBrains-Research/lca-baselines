```python
import nni
from nni.distiller import ModelSpeedup, QATQuantizer, DynamicLayerwiseDistiller, TaylorPruner, AGPPruner

# Fine-tune ResNet18 model on Cifar10 dataset for 30 epochs
# Evaluate accuracy

# Create teacher model by duplicating fine-tuned model

# Create pruner using TaylorPruner and AGPPruner targeting Conv2d operations with sparse ratio 0.5
pruner = TaylorPruner(target_sparsity=0.5, trainer=trainer, model=model)
pruner.compress()

# Create quantizer using QATQuantizer targeting Conv2d and BatchNorm2d operations with int8 quantization starting at 100th step

quantizer = QATQuantizer(target_modules=[nn.Conv2d, nn.BatchNorm2d], quant_types=['int8'], start_step=100)
quantizer.compress()

# Create distiller using DynamicLayerwiseDistiller targeting Conv2d operations with mean squared error method and teacher model
distiller = DynamicLayerwiseDistiller(teacher_model=teacher_model, distillation_loss='mse', target_modules=[nn.Conv2d])
distiller.compress()

# Compress model using distiller for 60 iterations of 100 steps each

# Speed up model using ModelSpeedup and masks from pruner
speedup = ModelSpeedup(model=model, pruner=pruner)
speedup.speedup()

# Evaluate accuracy of compressed model

# Simulate quantization by updating calibration configuration of model using Quantizer class

# Evaluate accuracy of compressed and quantized model
```