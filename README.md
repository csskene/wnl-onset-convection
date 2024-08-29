[![DOI](https://zenodo.org/badge/667766241.svg)](https://zenodo.org/doi/10.5281/zenodo.13284472)

# Weakly nonlinear analysis of the onset of convection in rotating spherical shells

This repository contains companion code for the paper 'Weakly nonlinear analysis of the onset of convection in rotating spherical shells' by C. S. Skene and S. M. Tobias (in review). A preprint is available [here](https://arxiv.org/abs/2408.15603).

## Available code
1. *critical_Rayleigh.py* finds the critical Rayleigh number for the onset of convection using optimisation techniques.
2. *wnl_coefficients.py* computes the coefficients of the weakly nonlinear amplitude equation.
3. *read_data.py* merges data files created by computing the weakly nonlinear coefficients at different paramters.
4. *convection.py* runs the full nonlinear equations starting from the unstable mode in order to validate the weakly nonlinear description.

## Citation
If this code is useful for your research, please cite
```
 @misc{skene2024,
    title={Weakly nonlinear analysis of the onset of convection in rotating spherical shells}, 
    author={Skene, Calum S. and Tobias, Steven M.},
    year={2024},
    eprint={2408.15603},
    archivePrefix={arXiv},
    primaryClass={physics.flu-dyn},
    url={https://arxiv.org/abs/2408.15603}, 
 }
```

## Acknowledgements
This work was undertaken on ARC4, part of the High Performance Computing facilities at the University of Leeds, UK. We would like to acknowledge Emmanuel Dormy and Andrew Soward for enlightening conversations. We acknowledge partial support from a grant from the Simons Foundation (Grant No. 662962, GF). We would also like to acknowledge support of funding from the European Union Horizon 2020 research and innovation programme (grant agreement no. D5S-DLV-786780).
