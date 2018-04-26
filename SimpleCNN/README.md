## SimpleCNN

This notebook contains classes to
- get the training and test data
- put it in dataframes for easy querying, etc.
- preprocess it for input to a Keras model
- define, train, and generate predictions via a simple Keras CNN
- save/load the CNN's weights
- generate predictions using the model
- post-process the predictions to clean them
- generate the run-length-encoded predictions file in Kaggle format
- compute the Kaggle score from a predictions file and the ground truth file

Several aspects of this may be useful to the team generally:
- The CNN can be quickly trained and tested, so it provides a useful testbed for testing different pre- and post-processing steps and data augmentation strategies
  - The basic CNN trains in just 15 minutes on my Macbook, so it's practical to do a lot of experimentation with. Even in that simple configuration, it gets a Kaggle score of 0.159. Not awesome, obviously, but high enough to show it's doing something meaningful. The Kaggle score class can output the ground truth and predicted mask images, and visual inspection shows a number of them are pretty good, so it really is doing something real. 
  
- The Kaggle scorer takes Kaggle files as input, so it's model method agnostic. Everyone should be able to use it to score their results. 
  - A couple of things to note about the Kaggle run-length-encodings: 1) the pixel locations are 1-based, and 2) the run-length-encoding is done in "Fortran" order: by columns, then rows. My Kaggle scoring class assumes it's getting input in that format.

The CNN also gives us an additional point of comparison beyond U-Net and Mask-RCNN that we can include in our project report.
