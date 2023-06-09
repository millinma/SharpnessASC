# K-FAC: Kronecker-Factored Approximate Curvature

## Changes to KFAC-Pytorch Repo

GitHub Link: https://github.com/ntselepidis/KFAC-Pytorch
Fork of: https://github.com/alecwangcq/KFAC-Pytorch\

-   Renamed from _KFAC-Pytorch_ to _KFACPytorch_.
-   Created _\_\_init\_\_.py_ to use as a Module.
-   Removal of _main.py_, _grid_search.py_, _modules.sh_, _plot.py_, _train_cifar_, _train_mnist.py_, _train_toy_, _requirements.txt_ and _models/_ as they are not part of the core functionality.
-   Removal of _utils/get_args.py_, _utils/log_utils.py_, _utils/lr_scheduler_utils.py_, _utils/network_utils.py_, _utils/optim_utils.py_ as they are not part of the core functionality.
-   Add optional print of KFAC layers.
-   Surpress User warnings (these will be removed in future torch versions!):
    -   non-full backward hook
    -   torch.symeig

## References

Please consider citing the following papers for K-FAC:

```
@inproceedings{martens2015optimizing,
  title={Optimizing neural networks with kronecker-factored approximate curvature},
  author={Martens, James and Grosse, Roger},
  booktitle={International conference on machine learning},
  pages={2408--2417},
  year={2015}
}

@inproceedings{grosse2016kronecker,
  title={A kronecker-factored approximate fisher matrix for convolution layers},
  author={Grosse, Roger and Martens, James},
  booktitle={International Conference on Machine Learning},
  pages={573--582},
  year={2016}
}
```

and for E-KFAC:

```
@inproceedings{george2018fast,
  title={Fast Approximate Natural Gradient Descent in a Kronecker Factored Eigenbasis},
  author={George, Thomas and Laurent, C{\'e}sar and Bouthillier, Xavier and Ballas, Nicolas and Vincent, Pascal},
  booktitle={Advances in Neural Information Processing Systems},
  pages={9550--9560},
  year={2018}
}
```

For two-level K-FAC please cite:

```
@misc{tselepidis2020twolevel,
  title={Two-Level K-FAC Preconditioning for Deep Learning},
  author={Nikolaos Tselepidis and Jonas Kohler and Antonio Orvieto},
  year={2020},
  eprint={2011.00573},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## Contact

If you have any questions or suggestions, please feel free to contact me via email at ntselepidis@student.ethz.ch.
