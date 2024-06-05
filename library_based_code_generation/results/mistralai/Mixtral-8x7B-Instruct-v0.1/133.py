 ```python
import fate_flow
from fate_flow.component import DataIO, Transform, Intersect, Evaluate, HeteroNN
from fate_flow.setting import JobParameter
from fate_flow.job_utils import main as job_main

def create_pipeline(config_file):
    job_param = JobParameter(config_file)

    data_io_guest = DataIO(job_param=job_param, role='guest')
    data_io_host = DataIO(job_param=job_param, role='host')

    transform_guest = Transform(job_param=job_param, role='guest')
    transform_host = Transform(job_param=job_param, role='host')

    intersect = Intersect(job_param=job_param)

    hetero_nn = HeteroNN(job_param=job_param,
                         role='guest',
                         bottom_model_guest='logistic',
                         bottom_model_host='logistic',
                         interactive_layer='fi_0.5_0.3',
                         top_model='logistic',
                         epochs=10,
                         learning_rate=0.01,
                         batch_size=256,
                         callbacks={'early_stopping': {'monitor': 'val_loss', 'patience': 3}})

    evaluate = Evaluate(job_param=job_param, role='guest')

    data_io_guest.set_input(intersect.get_output('data'))
    data_io_host.set_input(intersect.get_output('data'))

    transform_guest.set_input(data_io_guest.get_output('data'))
    transform_host.set_input(data_io_host.get_output('data'))

    intersect.set_input(transform_guest.get_output('data'), transform_host.get_output('data'))

    hetero_nn.set_input(intersect.get_output('data'))
    evaluate.set_input(hetero_nn.get_output('data'))

    hetero_nn.compile()
    hetero_nn.fit()

    print(hetero_nn.model.summary())

if __name__ == '__main__':
    main_func = job_main(create_pipeline)
    main_func()
```