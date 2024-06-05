``python
import argparse
import sirf
import numpy as np
from sirf.Logging import log
from sirf.ObjectiveFunctions import PoissonLogLikelihoodObjectiveFunction
from sirf.Acquisition import AcquisitionModel
from sirf.Image import Image

def truncate_image(image, diameter):
"""
Truncate the image to a cylinder of the specified diameter.
"""
x, y, z = image.getDimensions()
cylinder = np.zeros((x, y, z))
x_center, y_center = x // 2, y // 2
for i in range(x):
for j in range(y):
if (i - x_center)**2 + (j - y_center)**2 <= (diameter/2)**2:
cylinder[i, j, :] = image[i, j, :]
return Image(cylinder)

def main():
parser = argparse.ArgumentParser(description='Perform a few steps of steepest ascent for Poisson log-likelihood objective function.')
parser.add_argument('--engine', type=str, help='Reconstruction engine module to import from sirf library')
parser.add_argument('--raw-data-file', type=str, help='Raw data file')
parser.add_argument('--data-path', type=str, help='Path to data files')
parser.add_argument('--num-steps', type=int, default=5, help='Number of steepest descent steps')
parser.add_argument('--locally-optimal', action='store_true', help='Use locally optimal steepest ascent')
parser.add_argument('--verbose', action='store_true', help='Verbosity')
parser.add_argument('--no-plots', action='store_true', help='Do not show plots')
args = parser.parse_args()

try:
engine = __import__('sirf.' + args.engine)
except ImportError as e:
log.error(f'Could not import engine module: {e}')
return

acq_model = AcquisitionModel()
acq_model.read(args.raw_data_file, args.data_path)

filter_func = lambda x: truncate_image(x, acq_model.getImageDiameter())

initial_image = Image(np.zeros(acq_model.getImage().getDimensions()))

objective_func = PoissonLogLikelihoodObjectiveFunction(acq_model, filter_func)

for i in range(args.num_steps):
try:
if args.locally_optimal:
gradient = objective_func.locallyOptimalSteepestDescent(initial_image)
else:
gradient = objective_func.steepestDescent(initial_image)
except Exception as e:
log.error(f'Steepest descent failed: {e}')
return

initial_image += gradient

if not args.no_plots:
# Plotting code here

if args.verbose:
print(f'Performed {args.num_steps} steps of steepest ascent.')

if __name__ == '__main__':
main()
```