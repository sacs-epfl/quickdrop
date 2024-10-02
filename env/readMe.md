The folder stores the Federated Learning environments you created. 
We also provide an affine tool *eval_allocation.py* to help you 
visualize the interior contents and details.

How to use the tool?

`python eval_allocation.py --strategy [Your allocation strategy] --scenario [Full name] --class_num [Number of classes you want to visualize]`

example:

`python eval_allocation.py --strategy quickdrop-affine --scenario affine-cifar10-seed42-u20-alpha0.1-scale0.01 --class_num 10`