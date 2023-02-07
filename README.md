# AL-PINNs
Official code for AL-PINNs: Augmented Lagrangian relaxation method for Physics-Informed Neural Networks

Every parameter is set to its default value if not specified.

python3 Helmholtz_AL-PINNs.py --model=[model] --beta=[beta] --lr=[learning rate] --lbd_lr=[learning rate for lambda] --EPOCH=[training epoch] --ordinal=[cuda device]
python3 Helmholtz_vanilla_PINNs.py --model=[model] --beta=[beta] --lr=[learning rate] --EPOCH=[training epoch] --ordinal=[cuda device]
python3 Helmholtz_Lagrange_Multiplier.py --model=[model] --lr=[learning rate] --lbd_lr=[learning rate for lambda] --EPOCH=[training epoch] --ordinal=[cuda device]

python3 Klein-Gordon_AL-PINNs.py --model=[model] --beta=[beta] --lr=[learning rate] --lbd_lr=[learning rate for lambda] --EPOCH=[training epoch] --ordinal=[cuda device]
python3 Klein-Gordon_vanilla_PINNs.py --model=[model] --beta=[beta] --lr=[learning rate] --EPOCH=[training epoch] --ordinal=[cuda device]

python3 Viscous_Burgers_AL-PINNs.py --model=[model] --beta=[beta] --lr=[learning rate] --lbd_lr=[learning rate for lambda] --EPOCH=[training epoch] --ordinal=[cuda device]
python3 Viscous_Burgers_vanilla_PINNs.py --model=[model] --beta=[beta] --lr=[learning rate] --EPOCH=[training epoch] --ordinal=[cuda device]
