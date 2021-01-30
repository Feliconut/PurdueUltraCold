# ULTRACOLD ML CALIBRATOR PROJECT

## Goal

Instead of using a large ensemble of atomic images to directly obtain the OTF of the imaging system, which could be time-consuming, we develop a ML model to correctly **predict the imaging system parameters using just one single image**, thus greatly increasing the efficiency of our system calibration procedure.

## Files

`DATA/<ID>` represents a __observation batch__ using the same configuration, so all images under one observation batch have the same OTF and hence the same fitting parameters.

`fitting` contains a set of code that can carry out the traditional fitting process.

## Resources

Shared note: https://roamresearch.com/#/app/Feliconut/page/6GPf5dv1J