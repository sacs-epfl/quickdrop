This folder includes the artifact evaluation required codes and running results. Since we also have uploaded the checkpoints and the affine datasets (QuickDrop Algorithm), the computation is not intensive if you decide to test on our checkpoints. This means you can test the codes on a device without gpus.

We include some running results, which may takes hours if use cpu:
1. [QuickDrop-ConvNet-cifar10-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-scale0.01-le5.pth](check_point%2FQuickDrop-ConvNet-cifar10-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-scale0.01-le5.pth) [CIFAR10 QuickDrop Algorithm]
2. [QuickDrop-ConvNet-mnist-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-scale0.01-le5.pth](check_point%2FQuickDrop-ConvNet-mnist-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-scale0.01-le5.pth)[MNIST QuickDrop Algorithm]
3. [affine-cifar10-seed42-u20-alpha0.1-scale0.01](..%2Fenv%2Fquickdrop-affine%2Faffine-cifar10-seed42-u20-alpha0.1-scale0.01) [CIFAR10 QuickDrop Algorithm's Affine]
4. [affine-mnist-seed42-u20-alpha0.1-scale0.01](..%2Fenv%2Fquickdrop-affine%2Faffine-mnist-seed42-u20-alpha0.1-scale0.01) [MNIST QuickDrop Algorithm's Affine]


It now includes the code to reproduce the similar result and an extensive of Figure 2 in the original paper. These reproductions are folders:
1. [fig2_with_recovering](fig2_with_recovering): similar results in the Figure 2.
2. [fig2_without_recovering](fig2_without_recovering): a proper unlearning rate hardly degrades other class accuracies.

Each runs on the 2 datasets in total of 4 figures, which are stored in [result_1](fig2_with_recovering%2Fresult) and [result_2](fig2_without_recovering%2Fresult). 
You can run the codes [code_1](fig2_with_recovering%2Fcode) and [code_2](fig2_without_recovering%2Fcode). The command lines have been
included in their folders if computation device is available, or you can directly check the running results on our device. Our new running platform is an i9-13900K CPU and a RTX3090.

Thanks for recognizing our work.
