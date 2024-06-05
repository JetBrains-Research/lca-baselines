import os
import json
from tqdm import tqdm


def extract_workflows(json_folder, workflow_folder):

    dp_num = 0
    for filename in tqdm(os.listdir(json_folder)):
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder, filename)

            with open(json_path, 'r') as json_file:
                data = json.load(json_file)

            workflow_data = data.get('workflow')
            difficalty = data.get('difficulty')
            if workflow_data is not None and difficalty > 0:
                dp_num += 1
                yaml_filename = os.path.splitext(filename)[0] + '.yaml'
                yaml_path = os.path.join(workflow_folder, yaml_filename)

                with open(yaml_path, 'w') as f:
                    f.write(workflow_data)#, default_flow_style=False

    print(f"{dp_num} datapoints processed")

def write_workflows(json_folder, workflow_folder):

    for filename in os.listdir(workflow_folder):
        if filename.endswith('.yaml'):
            json_path = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(json_folder, json_path)
            workflow_path = os.path.join(workflow_folder, filename)

            with open(workflow_path, 'r') as f:
                workflow_data = f.read()

            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)

            json_data['workflow'] = workflow_data

            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file)

if __name__ == "__main__":
    dataset_folder = '/mnt/data/galimzyanov/data/LCA/HF_dataset/lca-ci-fixing_edited'
    workflow_folder = '/mnt/data/galimzyanov/data/LCA/HF_dataset/workflows'
    # extract_workflows(dataset_folder, workflow_folder)
    write_workflows(dataset_folder, workflow_folder)
