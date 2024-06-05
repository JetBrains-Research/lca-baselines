```python
import logging
import json
import gunpowder as gp

logging.basicConfig(level=logging.INFO)

def train_model(num_iterations):
    array_keys = {
        'RAW': gp.ArrayKey('RAW'),
        'GT_LABELS': gp.ArrayKey('GT_LABELS'),
        'GT_AFFINITIES': gp.ArrayKey('GT_AFFINITIES'),
        'LOSS_WEIGHTS': gp.ArrayKey('LOSS_WEIGHTS'),
        'PRED_AFFINITIES': gp.ArrayKey('PRED_AFFINITIES'),
        'LOSS_GRADIENT': gp.ArrayKey('LOSS_GRADIENT')
    }

    with open('config.json', 'r') as f:
        config = json.load(f)

    input_size = gp.Coordinate(config['input_size'])
    output_size = gp.Coordinate(config['output_size'])

    request = gp.BatchRequest()
    request.add(array_keys['RAW'], input_size)
    request.add(array_keys['GT_LABELS'], output_size)
    request.add(array_keys['GT_AFFINITIES'], output_size)
    request.add(array_keys['LOSS_WEIGHTS'], output_size)

    snapshot_request = gp.BatchRequest()
    snapshot_request[array_keys['PRED_AFFINITIES']] = request[array_keys['GT_AFFINITIES']]
    snapshot_request[array_keys['LOSS_GRADIENT']] = request[array_keys['GT_AFFINITIES']]

    pipeline = (
        gp.Hdf5Source(
            'train.hdf',
            datasets={
                array_keys['RAW']: 'volumes/raw',
                array_keys['GT_LABELS']: 'volumes/labels'
            }
        ) +
        gp.Normalize(array_keys['RAW']) +
        gp.RandomLocation() +
        gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2]) +
        gp.ElasticAugment([40, 40, 40], [0.25, 0.25, 0.25], [0, math.pi/2.0], prob_slip=0.05, prob_shift=0.05, max_misalign=25) +
        gp.IntensityAugment(array_keys['RAW'], 0.9, 1.1, -0.1, 0.1) +
        gp.GrowBoundary(array_keys['GT_LABELS'], steps=1, only_xy=True) +
        gp.AddAffinities(
            affinity_neighborhood=[
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ],
            labels=array_keys['GT_LABELS'],
            affinities=array_keys['GT_AFFINITIES']
        ) +
        gp.BalanceLabels(
            array_keys['GT_AFFINITIES'],
            array_keys['LOSS_WEIGHTS']
        ) +
        gp.PreCache(
            cache_size=5,
            num_workers=10
        ) +
        gp.Train(
            'train_net',
            optimizer=optimizer,
            loss=loss,
            inputs={
                'input': array_keys['RAW']
            },
            outputs={
                'output': array_keys['PRED_AFFINITIES']
            },
            gradients={
                'output': array_keys['LOSS_GRADIENT']
            }
        ) +
        gp.Snapshot(
            output_filename='batch_{iteration}.hdf',
            dataset_names={
                array_keys['RAW']: 'volumes/raw',
                array_keys['GT_LABELS']: 'volumes/labels',
                array_keys['PRED_AFFINITIES']: 'volumes/pred_affinities',
                array_keys['LOSS_GRADIENT']: 'volumes/loss_gradient'
            },
            additional_request=snapshot_request
        ) +
        gp.PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with gp.build(pipeline):
        for i in range(num_iterations):
            pipeline.request_batch(request)
    print("Training finished.")

if __name__ == "__main__":
    train_model(10000)
```