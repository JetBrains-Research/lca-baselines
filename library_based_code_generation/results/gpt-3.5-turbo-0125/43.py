import numpy as np
import matplotlib.pyplot as plt

# Read precipitation field data
precipitation_data = read_data()

# Upscale to 2 km resolution
upscaled_data = upscale_data(precipitation_data, resolution=2)

# Convert to rain rate
rain_rate_data = convert_to_rain_rate(upscaled_data)

# Log-transform the data
log_transformed_data = np.log(rain_rate_data)

# Handle missing values
cleaned_data = handle_missing_values(log_transformed_data)

# Estimate motion field
motion_field = estimate_motion_field(cleaned_data)

# Perform ensemble nowcast using STEPS approach
ensemble_nowcast = perform_steps_nowcast(cleaned_data, motion_field)

# Back-transform nowcast to rain rates
back_transformed_data = np.exp(ensemble_nowcast)

# Plot some realizations
plot_realizations(back_transformed_data)

# Verify probabilistic forecasts
roc_curve = calculate_roc_curve(back_transformed_data)
reliability_diagrams = calculate_reliability_diagrams(back_transformed_data)
rank_histograms = calculate_rank_histograms(back_transformed_data)