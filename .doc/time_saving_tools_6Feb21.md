# Proposal of Time-Saving Tools for Data Preparation

The objective of this pair of tools is to minimize the number of data samples for which the fitting success of the Noise Power Spectrum module needs to be manually checked in the immediate future and to create a tool that can be used in future sample preparation to bypass human intervention altogether and save meaningful time.

## Tool One: GUI driven data labelling
  
    This tool would go through an image folder in a selected directory, print the experimental and fitted noise power spectrum 
    utilizing fplot_NPS_ExpAndFit and necessary reduncancies from NPSmethods.py. It would then present the user with three buttons:
    Good Fit, Bad Fit, or Save and Quit. On Good or Bad press, it would create an array containing the experimental and fitted 
    imaging response functions and either a one or a zero depending on whether Good Fit or Bad Fit was pressed. It would then repeat 
    the process for all folders in the selected directory, appending the data to the array each time. Upon completion or Save and
    Quit press, it would exit the loop (optionally, it might also reduce the array down to a smaller array containing an equal 
    number of one and zero events) and save the array to a pickle file.
  
## Tool Two: CNN with binary classification

    This tool would open the pickle file created by tool one and separate into a 3D tensor of the experimental and fitted imaging
    response functions and a 1D tensor of the one and zero truth labels. It would then train a CNN to distinguish using the imaging 
    response functions, whether the fit is good or bad.
   
## Modifications to existing code

    The DataPrep notebook would then be modified to include passing the experimental and fitted imaging response functions through
    the CNN and only append the optical density image and calculated parameters in the event of the CNN labeling it as a good fit.
