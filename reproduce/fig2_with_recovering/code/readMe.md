Execute the following commands to obtain the similar results.
Make sure you have activated the conda environment.

`python reproduce_single_unlearning.py --checkpoint_path ../../check_point/QuickDrop-ConvNet-mnist-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-scale0.01-le5.pth --dataset MNIST --env affine-mnist-seed42-u20-alpha0.1-scale0.01 --forgetting_rate 0.01  --learning_rate 0.01`\
`python reproduce_single_unlearning.py --checkpoint_path ../../check_point/QuickDrop-ConvNet-cifar10-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-scale0.01-le5.pth --dataset CIFAR10 --env affine-cifar10-seed42-u20-alpha0.1-scale0.01 --forgetting_rate 0.004 --learning_rate 0.02`

If you don't have a gpu:

`python reproduce_single_unlearning.py --device cpu --checkpoint_path ../../check_point/QuickDrop-ConvNet-mnist-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-scale0.01-le5.pth --dataset MNIST --env affine-mnist-seed42-u20-alpha0.1-scale0.01 --forgetting_rate 0.01  --learning_rate 0.01`\
`python reproduce_single_unlearning.py --device cpu --checkpoint_path ../../check_point/QuickDrop-ConvNet-cifar10-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-scale0.01-le5.pth --dataset CIFAR10 --env affine-cifar10-seed42-u20-alpha0.1-scale0.01 --forgetting_rate 0.004 --learning_rate 0.02`