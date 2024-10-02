This project separates a Federated Unlearning (FU) training and unlearning process into: [1] training environment, [2] server and client behavior definition (OOP), and [3] complex assembly.
The motivation is to grant researchers the control and extensiveness of Federated Learning (FL). This exterior readMe (this one) file only describes the general usage of the following folders. You can further access these folders to view the detailed readMe file if interested.

[1], [2] include:
1. env_generator: used to generate federated learning training environment i.e. iid, dilichilet etc.
2. FedAvg: FedAvg server and client, which is the vanilla version of federated learning research.
3. FedNaiveRetrain: naive retrain server and client.
4. FedRecover: alias of FedNaiveRetrain.
5. FedSGA: server and client supporting stochastic gradient ascent (SGA).
6. **FedQuickDrop: our work**.
7. utils: datasets, models, and others.

[3] includes:
1. merit: further verification methods i.e. logits distribution described in the Kaggle NIPS23 Machine Unlearning Competition.
2. **reproduce: artifact evaluation. (If you want to quickly evaluate the work, go to this folder directly.)**
---

If you are not familiar with the conda/pytorch environment installation, we provide a tutorial [conda_env](conda_env) for you reference.

---
How to use the architecture? 
1. generate FL env using methods in the env_generator.
2. choose or customize servers and clients.
3. assemble your own experiment.
---