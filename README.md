# learnergy4video: Energy-based Machine Learners for Video

[![Latest release](https://img.shields.io/github/release/MateusRoder/learnergy4video.svg)](https://github.com/MateusRoder/learnergy4video/releases)
[![Open issues](https://img.shields.io/github/issues/MateusRoder/learnergy4video.svg)](https://github.com/MateusRoder/learnergy4video/issues)
[![License](https://img.shields.io/github/license/MateusRoder/learnergy4video.svg)](https://github.com/MateusRoder/learnergy4video/blob/master/LICENSE)

## Welcome to learnergy4video.

Did you ever reach a bottleneck in your computational experiments? Are you tired of implementing your own techniques? If yes, learnergy4video is the real deal! This package provides an easy-to-go implementation of energy-based machine learning algorithms for video domain. From big datasets to fully-customizable models, from internal functions to external communications, we will foster all research related to energy-based machine learning.

Use learnergy4video if you need a library or wish to:

* Create your energy-based machine learning algorithm;
* Design or use pre-loaded learners;
* Mix-and-match different strategies to solve your problem;
* Because it is incredible to learn things.


learnergy4video is compatible with: **Python 3.6+**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.

---

## Citation

If you use learnergy4video to fulfill any of your needs, please cite us:

```BibTex
@misc{roder2020learnergy4video,
    title={learnergy4video: Energy-based Machine Learners for Video Domain},
    author={Mateus Roder and Gustavo Henrique de Rosa and João Paulo Papa},
    year={202X},
    eprint={---},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

---

## Getting started: 60 seconds with learnergy4video

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, choose your subpackage, and follow the example. We have high-level examples for most of the tasks we could think.

Alternatively, if you wish to learn even more, please take a minute:

learnergy4video is based on the following structure, and you should pay attention to its tree:

```yaml
- learnergy4video
    - core
        - dataset
        - model
    - math
        - scale
    - models
        - binary (keep it original, videos cannot be binarized)
            - conv_rbm 
            - discriminative_rbm (needs the portability to continuous input)
            - dropout_rbm
            - e_dropout_rbm (needs the portability to continuous input)
            - rbm
        - real            
            - gaussian_conv_rbm
            - gaussian_rbm
            - dropout_grbm (to implement)
            - e_dropout_grbm (to implement)
            - sigmoid_rbm
            - spec_conv_rbm
        - stack
            - dbn
            - residual_dbn (needs the portability to video)
            - conv_dbn
            - spec_conv_dbn
            - dbm
            - spec_dbm (testing...)
    - utils
        - constants
        - exception
        - logging
        - collate function
        - ucf loader
        - hmdb loader
    - visual
        - image
        - metrics
        - tensor
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Math

Just because we are computing stuff, it does not means that we do not need math. Math is the mathematical package, containing low-level math implementations. From random numbers to distributions generation, you can find your needs on this module.

### Models

This is the heart. All models are declared and implemented here. We will offer you the most fantastic implementation of everything we are working with. Please take a closer look into this package.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing over and over again.

### Visual

Everyone needs images and plots to help visualize what is happening, correct? This package will provide every visual-related method for you. Check a specific image, your fitness function convergence, plot reconstructions, weights, and much more.

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, learnergy4video will be the one-to-go package that you will need, from the very first installation to the daily-tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

```bash
pip install learnergy4video
```

Alternatively, if you prefer to install the bleeding-edge version, please clone this repository and use:

```bash
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here, you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or mateus.roder@unesp.br and gustavo.rosa@unesp.br.

---
