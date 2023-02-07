# AL-PINNs
### Official code for:

### - [AL-PINNs: Augmented Lagrangian relaxation method for Physics-Informed Neural Networks](https://arxiv.org/abs/2205.01059)

# Files 
\* : Helmholtz, Klein-Gordon, Viscous_Burgers
### *_AL-PINNs.py : AL-PINNs implementation for each benchmark equation<br/> 
### *_vanilla_PINNs.py : Vanilla PINNs implementation for each benchmark equation

# Usages
### Every parameter is set to its default value if not specified.

\* : Helmholtz, Klein-Gordon, Viscous_Burgers
#### python3 *_AL-PINNs.py --model=[model] --beta=[beta] --lr=[learning rate] --lbd_lr=[learning rate for lambda] --EPOCH=[training epoch] --ordinal=[cuda device]<br/> 
#### python3 *_vanilla_PINNs.py --model=[model] --beta=[beta] --lr=[learning rate] --EPOCH=[training epoch] --ordinal=[cuda device]
