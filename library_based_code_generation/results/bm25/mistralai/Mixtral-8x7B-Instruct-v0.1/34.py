 ```python
import sys
import argparse
from fate_flow.bundle import BundleServiceClient
from fate_flow.settings import Settings
from fate_flow.entity.data_flow import DataFlow
from fate_flow.entity.model_file import ModelFile
from fate_flow.entity.model_statistics import ModelStatistics
from fate_flow.entity.model_eval import ModelEvaluate
from fate_flow.entity.model_meta import ModelMeta
from fate_flow.entity.table_meta import TableMeta
from fate_flow.entity.table_statistics import TableStatistics
from fate_flow.entity.data_profile import DataProfile
from fate_flow.entity.data_info import DataInfo
from fate_flow.entity.role_info import RoleInfo
from fate_flow.entity.job_graph import JobGraph
from fate_flow.entity.job_parameter import JobParameter
from fate_flow.entity.job_dag import JobDag
from fate_flow.component.data_preprocess.dataio import DataIO
from fate_flow.component.feature_engineering.feature_binning import FeatureBinning
from fate_flow.component.feature_engineering.feature_scaling import FeatureScaling
from fate_flow.component.feature_engineering.feature_selection import FeatureSelection
from fate_flow.component.feature_engineering.feature_engineering import FeatureEngineering
from fate_flow.component.model_train.logistic_regression import LogisticRegression
from fate_flow.component.model_evaluation.model_eval import ModelEvaluation
from fate_flow.component.model_analysis.model_statistics import ModelStatisticsComponent

def create_pipeline(config_file=None):
    if config_file is None:
        config_file = 'config.json'

    # Initialize data and variable
    init_data_and_variable = DataIO(
        init_data=TableMeta(
            table_name="table_name",
            namespace="namespace",
            data_type="data_type",
            description="description",
            partition_file="partition_file",
            file_type="file_type",
            file_format="file_format",
            file_path="file_path",
            file_url="file_url",
            file_size="file_size",
            file_md5="file_md5",
            file_row_num="file_row_num",
            file_column_num="file_column_num",
            file_column_type="file_column_type",
            file_column_name="file_column_name",
            file_column_description="file_column_description",
            file_column_order="file_column_order",
            file_column_is_nullable="file_column_is_nullable",
            file_column_is_unique="file_column_is_unique",
            file_column_is_primary_key="file_column_is_primary_key",
            file_column_is_foreign_key="file_column_is_foreign_key",
            file_column_is_partition_key="file_column_is_partition_key",
            file_column_is_sort_key="file_column_is_sort_key",
            file_column_default_value="file_column_default_value",
            file_column_comment="file_column_comment",
            file_column_is_frozen="file_column_is_frozen",
            file_column_is_generated="file_column_is_generated",
            file_column_is_stored_as_text="file_column_is_stored_as_text",
            file_column_is_virtual="file_column_is_virtual",
            file_column_is_hidden="file_column_is_hidden",
            file_column_is_key="file_column_is_key",
            file_column_is_index="file_column_is_index",
            file_column_is_join_key="file_column_is_join_key",
            file_column_is_search_key="file_column_is_search_key",
            file_column_is_stats_key="file_column_is_stats_key",
            file_column_is_histogram_key="file_column_is_histogram_key",
            file_column_is_cluster_key="file_column_is_cluster_key",
            file_column_is_bloom_filter="file_column_is_bloom_filter",
            file_column_is_tokenized="file_column_is_tokenized",
            file_column_is_tokenized_with_ngram="file_column_is_tokenized_with_ngram",
            file_column_token_separator="file_column_token_separator",
            file_column_token_num="file_column_token_num",
            file_column_token_size="file_column_token_size",
            file_column_token_pattern="file_column_token_pattern",
            file_column_token_stopwords="file_column_token_stopwords",
            file_column_token_stemmer="file_column_token_stemmer",
            file_column_token_normalizer="file_column_token_normalizer",
            file_column_token_filter="file_column_token_filter",
            file_column_token_preserve_case="file_column_token_preserve_case",
            file_column_token_min_gram="file_column_token_min_gram",
            file_column_token_max_gram="file_column_token_max_gram",
            file_column_token_side="file_column_token_side",
            file_column_token_position="file_column_token_position",
            file_column_token_offset="file_column_token_offset",
            file_column_token_encoding="file_column_token_encoding",
            file_column_token_language="file_column_token_language",
            file_column_token_script="file_column_token_script",
            file_column_token_algorithm="file_column_token_algorithm",
            file_column_token_filter_language="file_column_token_filter_language",
            file_column_token_filter_script="file_column_token_filter_script",
            file_column_token_filter_algorithm="file_column_token_filter_algorithm",
            file_column_token_filter_stopwords="file_column_token_filter_stopwords",
            file_column_token_filter_stemmer="file_column_token_filter_stemmer",
            file_column_token_filter_normalizer="file_column_token_filter_normalizer",
            file_column_token_filter_filter="file_column_token_filter_filter",
            file_column_token_filter_preserve_case="file_column_token_filter_preserve_case",
            file_column_token_filter_min_gram="file_column_token_filter_min_gram",
            file_column_token_filter_max_gram="file_column_token_filter_max_gram",
            file_column_token_filter_side="file_column_token_filter_side",
            file_column_token_filter_position="file_column_token_filter_position",
            file_column_token_filter_offset="file_column_token_filter_offset",
            file_column_token_filter_encoding="file_column_token_filter_encoding",
            file_column_token_filter_language="file_column_token_filter_language",
            file_column_token_filter_script="file_column_token_filter_script",
            file_column_token_filter_algorithm="file_column_token_filter_algorithm",
            file_column_token_filter_filter="file_column_token_filter_filter",
            file_column_token_filter_preserve_case="file_column_token_filter_preserve_case",
            file_column_token_filter_min_gram="file_column_token_filter_min_gram",
            file_column_token_filter_max_gram="file_column_token_filter_max_gram",
            file_column_token_filter_side="file_column_token_filter_side",
            file_column_token_filter_position="file_column_token_filter_position",
            file_column_token_filter_offset="file_column_token_filter_offset",
            file_column_token_filter_encoding="file_column_token_filter_encoding",
