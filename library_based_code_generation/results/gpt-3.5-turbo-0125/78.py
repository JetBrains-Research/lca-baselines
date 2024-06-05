from fate_flow_client import FateFlowClient
from federatedml.feature.instance import Instance
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.feature.one_hot_encoder import OneHotEncoder
from federatedml.model_selection import train_test_split
from federatedml.linear_model.logistic_regression import LogisticRegression
from federatedml.secureprotol import Paillier
from federatedml.boosting.boosting_core import Boosting
from federatedml.evaluation.evaluation import Evaluation
from federatedml.util import consts

def main(config_file):
    # Read data
    guest_data = read_data_from_table(config_file['guest_table'])
    host_data = read_data_from_table(config_file['host_table'])
    
    # Transform data
    guest_data_transformed = transform_data(guest_data)
    host_data_transformed = transform_data(host_data)
    
    # Sample data
    guest_data_sampled = sample_data(guest_data_transformed)
    host_data_sampled = sample_data(host_data_transformed)
    
    # Feature binning
    feature_binning = QuantileBinning()
    guest_data_binned = feature_binning.fit_transform(guest_data_sampled)
    host_data_binned = feature_binning.transform(host_data_sampled)
    
    # One-hot encoding
    one_hot_encoder = OneHotEncoder()
    guest_data_encoded = one_hot_encoder.fit_transform(guest_data_binned)
    host_data_encoded = one_hot_encoder.transform(host_data_binned)
    
    # Logistic regression
    logistic_regression = LogisticRegression()
    logistic_regression.fit(guest_data_encoded)
    
    # Local baseline model
    local_baseline_model = LogisticRegression()
    local_baseline_model.fit(host_data_encoded)
    
    # Secure boosting
    secure_boosting = Boosting()
    secure_boosting.fit(guest_data_encoded, host_data_encoded)
    
    # Evaluation
    evaluation = Evaluation()
    evaluation_summary = evaluation.summary()
    print(evaluation_summary)

if __name__ == "__main__":
    main(config_file)