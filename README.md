# Extreme Value evolving Predictor - EVeP

The Extreme Value evolving Predictor (EVeP) is an evolving fuzzy-rule-based algorithm for online data streams. It offers a statistically well-founded approach to define the evolving fuzzy granules that form the antecedent and the consequent parts of the rules. The evolving fuzzy granules correspond to radial inclusion Weibull functions interpreted by the Extreme Value Theory as the limiting distribution of the relative proximity among the rules of the learning model. Regarding the parameters of the Takagi-Sugeno term at the consequent of the rules, the algorithm enhances the already demonstrated benefits of Multitask Learning by replacing a binary version with a fuzzy structural relationship among the rules. The pairwise similarity among the rules is automatically provided by the current interaction of the evolving fuzzy granules both at the antecedent and at the consequent parts of their corresponding rules. Several computational experiments, using artificial and real-world time series, attest to the dominating prediction performance of EVeP when compared to state-of-the-art evolving algorithms.

License
=======

This version of EVeP is released under the MIT license. (see LICENSE.txt).

Please cite EVeP in your publications if it helps your research:

...

Running
=======

EVeP was implemented with Python 3.6.

Prerequisites
-------------

- libMR - Library for Meta-Recognition and Weibull based calibration of SVMdata. Used to apply the methods founded on the Extreme Value Theory. Available at https://pypi.org/project/libmr/. More information can be found in Scheirer, W. J., Rocha, A., Micheals, R. J., & Boult, T. E. (2011). Meta-recognition: The theory and practice of recognition score analysis. IEEE transactions on pattern analysis and machine intelligence, 33(8), 1689-1695.

- MLflow - Open source platform to manage the ML lifecycle. Used to generate the results of the experiments. Available at https://mlflow.org/.

Project layout
--------------

Contributor
===========

Developed by Amanda O. C. Ayres under supervision of Prof. Fernando J. Von Zuben
