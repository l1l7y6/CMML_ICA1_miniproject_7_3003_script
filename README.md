# CMML3 ICA1 Mini-project7 - by 3003

This repository contains code for my CMML3 ICA1 mini-project on modelling the depolarising afterpotential (DAP) in oxytocin neurones using HypoMod.

## Main files
- `SpikeModPython.py` – entry point
- `spikemod.py` – main spiking model
- `spikepanels.py` – parameter panel definitions
- `HypoModPy/` – supporting HypoMod framework

## Modifications
- Added a simple DAP mechanism to the existing integrate-and-fire model
- Modified panel parameters to support DAP fitting
- Compared baseline and simple DAP fits for selected recordings

## Run
Run the program with:

```bash
python SpikeModPython.py
