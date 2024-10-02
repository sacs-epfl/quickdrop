The folder stores the allocation strategies. 
We provide dilichlet allocator here, whose running results will be stored in the *env*.
Please note that the running result cannot be identified unless you manually move it from the dataset subfolder, the process
is to avoid unwanted overwrite during training.

How to use the [allocator](dilichlet_allocator%2Fdilichlet_allocator.py)?

`cd dilichlet-allocator`

`python dilichlet_allocator --dataset_name [dataset_name] --num_clients [num_clients] --alpha [non-iid degree] --seed [random seed]`

example:

`python dilichlet_allocator --dataset_name CIFAR10 --num_clients 20 --alpha 0.1 --seed 42`