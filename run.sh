python main.py --dataset 'cifar10' --ood_rate 0.1 --ood_rate_1 0.5 --ood_rate_2 0.1 --ood_mix_rate 0.5 --id_rate 0.5 --scores 'energy' --ft_lr 0.001 --strategy 'random' --cortype 'gaussian_noise'  \
--budget 1000 --ft_weight 10 --aux_out_dataset 'places' --test_out_dataset 'places' --gpu 0

