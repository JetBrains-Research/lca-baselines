  Here is the Python code based on the given instructions:
```
import numpy as np
import matplotlib.pyplot as plt
from pysteps import nowcast_main_loop
from pysteps.utils import to_rainrate
from pysteps.utils import to_raindepth
from pysteps.utils import to_reflectivity
from pysteps.utils import _create_motion_field
from pysteps.utils import _convert_proj4_to_grid_mapping
from pysteps.utils import _convert_grid_mapping_to_proj4
from pysteps.utils import _create_data_univariate
from pysteps.utils import _create_data_multivariate
from pysteps.utils import DataModelError
from pysteps.utils import download_pysteps_data
from pysteps.utils import download_mrms_data
from pysteps.utils import test_to_rainrate

# Read precipitation field data
data = download_pysteps_data('precipitation_field_data')

# Upscale the data to 2 km resolution
data = data.reshape((data.shape[0] * 2, data.shape[1] * 2))

# Convert the data to rain rate
data = to_rainrate(data)

# Log-transform the data and handle missing values
data = np.log(data)
data[np.isnan(data)] = 0

# Estimate the motion field
motion_field = _create_motion_field(data)

# Perform an ensemble nowcast using the STEPS approach
nowcast = nowcast_main_loop(data, motion_field, timesteps=10)

# Back-transform the nowcast to rain rates
nowcast = to_rainrate(nowcast)

# Plot some of the realizations
plt.figure(figsize=(10, 10))
plt.imshow(nowcast[0], cmap='gray')
plt.title('Ensemble Nowcast')
plt.show()

# Verify the probabilistic forecasts using the ROC curve, reliability diagrams, and rank histograms
# ...
```
Note that this code is just a starting point and may need to be modified and expanded based on the specific requirements of the project. Additionally, the `download_pysteps_data` and `download_mrms_data` functions are not included in the code snippet provided, as they are not part of the `pysteps` library.