# Dataiku_task

## Modelling whether someone makes > $50,000 per year.
Using US census data to build the best possible model, and exploring which demographic factors are the most predictive.

## Method
Cleaning the data and engineering the best possible features. Transformation functions can be seen in a seperate .py file. Approach used:
1) Concatenating train and test data, then dropping all duplicate records
2) Converting age in to age ranges.
3) Eliminating Children from the model, as I think this will heavily bias the outcome. 
4) Treating any values labelled "Not in Universe" as missing.
5) Re-labelling salary as a binary variable.
6) Graphing frequency counts of all variables split by salary, to see which variables should be included in the model.
7) Encoding categoric variables
8) Upsampling the minority salary class
9) Creating a train-test split

