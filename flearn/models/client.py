import numpy as np

from flearn.utils.tf_utils import graph_size, process_grad



class Client(object):
    
    def __init__(self, id, group=None, dataset_name=None, model_name=None,
                 train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, model=None):
        self.model = model
        self.id = id
        self.group = group
        self.train_data = {k: np.array(v) for k, v in train_data.items()}
        self.eval_data = {k: np.array(v) for k, v in eval_data.items()}
        self.num_samples = len(self.train_data['y'])
        self.test_samples = len(self.eval_data['y'])
        self.model_name = model_name
        self.dataset_name = dataset_name

    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()

    def get_grads(self, model_len):
        '''get model gradient'''
        return self.model.get_gradients(self.train_data, model_len)

    def solve_grad(self):
        '''get model gradient with cost'''
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.train_data)
        comp = self.model.flops * self.num_samples
        bytes_r = self.model.size
        return ((self.num_samples, grads), (bytes_w, comp, bytes_r))

    def solve_inner(self, num_epochs=10, batch_size=7):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted  # communication costs
            (bytes_write here count for the full model, without couting the communication reduction by sparsification)
        '''

        bytes_w = self.model.size
        soln, comp = self.model.solve_inner(self.train_data, num_epochs, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)


    def train_error_and_loss(self):
        '''tests current model on local train_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        tot_correct, loss = self.model.test(self.train_data)
        return tot_correct, loss, self.num_samples


    def test(self):
        '''tests current model on local test_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        tot_correct, loss = self.model.test(self.eval_data)
        return tot_correct, self.test_samples