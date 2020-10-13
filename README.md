# Dataiku_task

## Modelling whether someone makes > $50,000 per year.
Using US census data from 1994-95 to build the best possible model, and exploring which demographic factors are the most predictive.

## Method
Cleaning the data and engineering the best possible features. Transformation functions can be seen in a seperate .py file. Approach taken:
1) Concatenating train and test data, then dropping all duplicate records
2) Converting age in to age ranges.
3) Classing anyone with under 12th grade education as Children, and removing from the model, otherwise this would heavily bias the outcome. 
4) Treating any values labelled "Not in Universe" as missing.
5) Re-labelling salary as a binary variable.
6) Graphing frequency counts of all variables split by salary, to see which variables should be included in the model.
7) Encoding categoric variables, with first value dropped.
8) Upsampling the minority salary class
9) Regularizing the data
10) Creating a train-test split
11) Applying different models, looking out for model accuracy and recall scores.
12) Analysing Logistic Regression coefficients 
13) Building a more powerful Random Forest Classifier 

## Modelling
I didn't get much improvement out of using tree-based methods, ultimately the Logistic Regression was more useful as I can easily view the model coefficients.
I compared accuracy & recall to determine which model to use.
Ultimately deciding on l2-Logistic Regression with upsampled and one-hot encoded categoric variables.

## Findings
Out of 96 variables, the ones most positively related to earning over $50,000 are:
1) Being aged above 40, particularly the 40 - 45 year group.
2) Having a job code = Executive or Manager
3) Being Male
4) Having a Bachelor or Master's degree

Lowest Negative coefficients are:
1) having tax status = non-filer. This also could be indicative of other things, such as being unemployed.
2) Working in Education or Retail.
3) Being married with civilian spouse present (which means other partner is in the army)



