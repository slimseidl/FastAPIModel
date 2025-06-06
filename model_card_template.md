# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained using the U.S. Census Income dataset. It predicts whether an individual's annual income exceeds $50,000 based on demographic features such as education, occupation, relationship, race, and native country. The model was implemented using scikit-learn and train_model.py.

## Intended Use

The model is intended to be used for deploying ML models via FastAPI, peforming performance evaluation across slices, and practicing reproducible ML workflows using GitHub and CI/CD.

## Training Data

The training data is a subset of the income dataset. It contains demographic and employment data such as age, working class, education, marital status, race, sex, etc.. The data was split into 80% training and 20% testing with a random state of 42 to ensure its reproductive ability.

## Evaluation Data

The test set contains 20% of the original dataset. It was processed using one hot encoder. The model was evaluated on slices of data and its unique values. 

## Metrics

The model used precision, recall, and F1 score for its performance metrics. Overall performance consisted of a precision of 0.7419, recall of 0.6384, and an F1 score of 0.6863. 

## Ethical Considerations

The model reflects biases present in the original dataset, which may include certain economic biases. Categorical features such as sex and race could introduce a difference in treatment between predictions. 

## Caveats and Recommendations

The model shouldn't be used for real-world classification until additional validation is confirmed. Categorical encoding can affect performance across different subgroups. 