**Attack => Adversarial, Data Poisoning**

---

> **Adversarial**

---

Consist of specially-crafted data points that are routed to the ML model to cause a faulty or wrong inference.

_Videos_

[Adversarial Attacks on Neural Networks - Bug or Feature?](https://www.youtube.com/watch?v=AOZw1tgD8dA)

[Adversarial Examples and Adversarial Training](https://www.youtube.com/watch?v=CIfsB_EYsVI)

_Papers_

[One Pixel Attack for Fooling Deep Neural Attack (https://ieeexplore.ieee.org/abstract/document/8601309)

[**Adversarial Examples Are Not Bugs, They Are Features**](http://gradientscience.org/adv/) (About Robust Features)

_Code_

[Robustness Package - Input Manipulation and Robustness of models](https://robustness.readthedocs.io/en/latest/)

---

> **Data poisoning**

---

Occur at training time and inject poisoned data points in the dataset.

Possible scenarios

Scraping images from the web

Harvesting system inputs (spam detector)

Federated learning & data collection

_Papers_

[Feature Collision](https://arxiv.org/abs/1804.00792) [\[Code\]](https://github.com/ashafahi/inceptionv3-transferLearn-poison) (Clean label poisoning attack)

[Backdoor Attack](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf) ( Fools models by imprinting a small number of training examples with a specific pattern (trigger) and changing their labels to a different target label )

[Bullseye Polytope](https://arxiv.org/abs/2005.00191) [\[Code\]](https://github.com/ucsb-seclab/BullseyePoison)

Targeted Poisoning

Label Flipping (Attacker is allowed to change the label of examples) Watermarking (Attacker perturbs the training image, not label, by superimposing a target image onto training images)

[Poisoning Benchmark](https://github.com/aks2203/poisoning-benchmark) [Paper](https://arxiv.org/abs/2006.12557)

_Code_

[**Adversarial Robustness Toolbox**](https://github.com/Trusted-AI/adversarial-robustness-toolbox) [Both for Attacks and Defenses]

---

> **Ensemble Method**

---

Ensemble learning is a machine learning technique that involves combining multiple models to improve the performance of a single model. The idea behind ensemble learning is to leverage the power of multiple models that are individually weak, but together can make better predictions than any single model alone.

_Techniques_ [Bagging, Boosting, Stacking]

Bagging: Training multiple models on different subsets of the data, and then combining their predictions by taking the average or majority vote.

Boosting: Sequentially training models on the same dataset, but each subsequent model focuses on the errors of the previous model.

Stacking: Training multiple models and using their predictions as inputs to a higher-level model.

**Papers**

[Popular Ensemble Methods: An Empirical Study](https://www.jair.org/index.php/jair/article/view/10239)

---

> **Ensemble Method Against Data Poisoning**

---

On the Robustness of Ensemble-Based Machine Learning Against Data Poisoning

**(** Evaluating the robustness of a hash-based Ensemble approach against data**)**

[On Collective Robustness of Bagging Against Data Poisoning](https://proceedings.mlr.press/v162/chen22k.html) [\[Code\]](https://media.icml.cc/Conferences/ICML2022/supplementary/chen22k-supp.zip)

---

> **Collective Robustness of Bagging Against Data Poisoning**

---

Bagging is a natural plug-and-play method with a high compatibility with various model architectures and training algorithms which is a commonly used method to avoid overfitting.

Some works have proved the **sample-wise robustness certificates** against the sample-wise attack but there is a white space in the **collective robustness certificates\*** against the global poisoning attack.

\*A collective robustness certificate is a way of proving that a classifier can make a certain number of predictions that are robust to adversarial perturbations of the input data.

This paper attempts to take the first step towards the collective certification for general bagging.

_Key Ideas_:

\> Binary Integer Linear Programming (BILP) problem to maximize the number of simultaneously changed predictions w.r.t. the given poison budget. To reduce the cost of solving the BILP problem, a decomposition strategy is devised.

\> Hash bagging to improve the robustness of vanilla bagging almost for free

**Datasets:**

[Bank (Moro et al., 2014)](https://www.sciencedirect.com/science/article/abs/pii/S016792361400061X) - [Dataset Characteristics](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#) - [Download Link](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/)

[Splice-2 comparative evaluation: Electricity pricing](https://citeseerx.ist.psu.edu/doc/10.1.1.43.9013) - [Description & Code](https://www.kaggle.com/datasets/yashsharan/the-elec2-dataset)

[Cifar-10](https://www.kaggle.com/c/cifar-10)

[FMNIST](https://github.com/zalandoresearch/fashion-mnist)
