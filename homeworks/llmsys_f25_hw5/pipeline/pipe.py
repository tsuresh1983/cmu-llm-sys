from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module, WithDevice

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN ASSIGN5_2_1
    num_clocks = num_batches + num_partitions - 1

    for k in range(num_clocks):
        schedule = []
        for i in range(num_batches):
            j = k - i
            if 0 <= j < num_partitions:
                schedule.append((i, j))
        yield schedule
        
    # raise NotImplementedError("Schedule Generation Not Implemented Yet")
    # END ASSIGN5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN ASSIGN5_2_2

        batches = list(x.split(self.split_size, dim=0))

        num_batches = len(batches)
        num_partitions = len(self.partitions)
        
        for schedule in _clock_cycles(num_batches, num_partitions):
          self.compute(batches, schedule)

        # for i, batch in enumerate(batches):
        #     batch.to(self.devices[-1])
        result = torch.cat(batches, dim=0)
        return result
        # raise NotImplementedError("Pipeline Parallel Not Implemented Yet")
        # END ASSIGN5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN ASSIGN5_2_2
        
        for i, j in schedule:
            batch = batches[i]
            partition = partitions[j]
            device = devices[j]
            batch = batch.to(device)
            from pipeline.partition import _retrieve_device
            if batch.device != _retrieve_device(partition):
                raise Exception("Batch and module are not on same device")
            task = Task(lambda b=batch, p=partition: p(b))
            self.in_queues[j].put(task)
            
        for i, j in schedule:
          success, result = self.out_queues[j].get()
          if not success:
              exc_info = result
              raise Exception(j, str(exc_info))

          task, batch = result
        #   batch.to(self.devices[-1])
          batches[i] = batch
          
        # raise NotImplementedError("Pipeline Parallel Not Implemented Yet")
        # END ASSIGN5_2_2

if __name__ == "__main__":
    batch_size = 16
    split_size = 2
    model = nn.Sequential(
        nn.Linear(3, 4).to('cuda:0'),
        WithDevice(nn.Sigmoid(), 'cuda:0'),
        nn.Linear(4, 5).to('cuda:0'),
        WithDevice(nn.Sigmoid(), 'cuda:0'),
    )
    
    x = torch.randn(batch_size, 3).to('cuda:0')
    y0 = model(x).to('cpu')

    # move the last two layer to another device
    model[-2] = model[-2].to('cuda:1')
    model[-1] = WithDevice(nn.Sigmoid(), 'cuda:1')
    pipe = Pipe(model, split_size=split_size)
    y1 = pipe(x).to('cpu')
    assert torch.allclose(y0, y1)