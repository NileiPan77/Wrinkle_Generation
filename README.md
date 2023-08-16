# Wrinkle_Generation

Implementation of Disney's [paper](https://studios.disneyresearch.com/app/uploads/2023/07/GraphBasedSynthesisForSkinMicroWrinkles.pdf). <img src=".\img\wrinkle_gen.png" alt="wrinkle_gen" style="zoom:50%;" />

Fig1. Adding artificial wrinkles and pores to a low res displacement texture.

## Current Progress

- **Python version of wrinkle generation finished, but not implemented as a tool for real time processing.**
- **C++ version in progress. Graph generation complete.**

## TODO

- Match c++'s wrinkle generation parameters and behavior to python version.
- Refactor c++'s implementation to use only primitive types, so it can be accelerated in GPU.
- Incorporate with a GUI to allow real time editing.

## Requirements

C++17, Boost Library, Opencv2
