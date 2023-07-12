# depressionDetection

Have taken up the dataset from kaggle as a problem statement.
Ran some  textpre-processing techniques such as 
lower case conversion,
removal of html and url tags,
stopwords removal,
chat words treatment,
replacing emojis with their corresponding meanings,
lemmatization.
Then converted the words into vectors using count vectorizer.
Then built a random forest classifier using the sklearn library.
These were the results of the used metrics:
Accuracy: 0.9375
Precision: 1.0
Recall: 0.5
F1-Score: 0.6666666666666666
In the end, built a basic flask web application for a front-end user exprience.
