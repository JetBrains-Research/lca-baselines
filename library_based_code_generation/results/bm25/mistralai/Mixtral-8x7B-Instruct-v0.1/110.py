 import argparse
import sys

import sirf.Engine as Engine
import sirf.ImageData as ImageData
import sirf.ImageDataProcessor as ImageDataProcessor
import sirf.ObjectiveFunction as ObjectiveFunction
import sirf.PoissonLogLikelihoodWithLinearModelForMeanAndProjData as PoissonLogLikelihoodWithLinearModelForMeanAndProjData
import sirf.TestSTIRObjectiveFunction as TestSTIRObjectiveFunction
import sirf.TruncateToCylinderProcessor as TruncateToCylinderProcessor
import sirf.AcquisitionData as AcquisitionData

def truncate_image(image, diameter):
processor = TruncateToCylinderProcessor.TruncateToCylinderProcessor(diameter)
return processor.process(image)

def steepest_ascent(engine_name, data_file, data_dir, num_steps, use_locally_optimal, verbose, plot):
try:
EngineModule = importlib.import\_module(engine\_name)
engine = EngineModule.Engine()

args = argparse.ArgumentParser()
args.add\_argument('–data\_file', type=str, default=data\_file, help='Path to the raw data file')
args.add\_argument('–data\_dir', type=str, default=data\_dir, help='Path to the data files')
args.add\_argument('–num\_steps', type=int, default=num\_steps, help='Number of steepest descent steps')
args.add\_argument('–use\_locally\_optimal', action=‘store\_true’, default=use\_locally\_optimal, help='Use locally optimal steepest ascent')
args.add\_argument('–verbose’, action=‘store\_true’, default=verbose, help='Verbosity')
args.add\_argument('–plot’, action=‘store\_true’, default=plot, help='Show plots')
options = args.parse\_args()

acquisition\_data = AcquisitionData.AcquisitionData.from\_acquisition\_data\_file(options.data\_dir, options.data\_file)

image = EngineModule.create\_uniform\_image(acquisition\_data.get\_geometry())
image = truncate\_image(image, acquisition\_data.get\_geometry().get\_xy\_size())

objective\_function = ObjectiveFunction.ObjectiveFunction(PoissonLogLikelihoodWithLinearModelForMeanAndProjData.PoissonLogLikelihoodWithLinearModelForMeanAndProjData(acquisition\_data))
engine.set\_objective\_function(objective\_function)

for i in range(options.num\_steps):
if options.use\_locally\_optimal:
engine.do\_locally\_optimal\_steepest\_ascent\_step()
else:
engine.do\_steepest\_ascent\_step()

if verbose:
print(‘Step: {0}, Value: {1}’.format(i, engine.get\_objective\_function\_value()))

except Exception as e:
print(‘Error: {0}’.format(e))
sys.exit(1)

if **name** == '**main**':
steepest\_ascent('sirf.PETEngine', 'data/projection\_data.h5', 'data/', 10, False, True, True)