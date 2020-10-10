# Disaster Pipeline #
Classify emergency messages (ML Pipeline Project for Udacity Data Scientist Nanodegree Program)


## Table of Contents ##
1.	[Motivation](#motivation)
2.	[Method](#method)
3.	[Data](#data)
4.	[Modeling Details](#modeling_details)
5.	[Discussion](#discussion)
6.	[File Descriptions](#file_descriptions)
7.	[Libraries/Packages](#libraries_packages)
8.	[Acknowledgements](#acknowledgements)
9.  [Licensing](#licensing)
10.  [Authors](#authors)


## Motivation<a name="motivation"></a> ##
Millions of messages are continually sent directly or posted on social media. Following a disaster (flood, fire, earthquake, etc.), authorities and aid agencies process these messages to identify those relevant to the disaster – perhaps one message in a thousand.
Aid organizations are often geared to meeting specific needs. Some specialize in food and water, some in clothing, others in medical supplies, and so on. Messages identified as being disaster-related must be routed to the appropriate organization depending upon the specific needs mentioned in the message.
The sheer volume of messages makes it impossible for staff to monitor, filter and route messages, especially in the hours immediately following a disaster – an automatic mechanism is needed. The objective of this project is to build a model that is capable of processing a message to determine whether it is disaster-related, and to classify it according to the kind of situation identified and/or help requested. This information can then be used to provide the appropriate response.


## Method<a name="method"></a> ##
A two-step method is used in the current project:
1. **Preparatory step:** the first step is to build a classifier model and have it ready for deployment. This is further divided into three steps:
    1. Extract-Transform-Load (ETL): this consists of extracting raw data from a source, transforming it into a form suitable for input to a model and loading the transformed data in a database. An ETL Pipeline is created to perform this task;
    2. Natural Language Processing (NLP): in this step, messages are read from the database. Each message is condensed by retaining only those words that are judged to be essential to the message’s meaning;
    3.	Machine Learning (ML): in this step, the condensed messages generated by the NLP step are used to train a model with the goal that the model is able to classify a new (“real live”) message belonging to one or more of several pre-identified categories (e.g., flood, earthquake, food, water, etc.)
2.	**Live deployment:** for purposes of this project, live deployment takes the form of typing a message into a web page. An application analyzes the message and displays the message categories on the web page.


## Data<a name="data"></a> ##
The curated data set used in this project was provided by [Figure Eight Inc](https://www.figure-eight.com) through [Udacity](http://www.udacity.com). The data consists of almost thirty thousand messages. Each message is labeled as belonging (or not) to each of thirty six categories.


## Modeling Details<a name="modeling_details"></a> ##
1.	**ETL Pipeline:**
    1.	*Extract:* Read messages and their pre-tagged categories from the respective csv files;
    2.	*Transform:*
        1.	Merge the two data sets;
        2.	Split the single string of categories into individual categories;
        3.	Remove duplicate rows;
        4.	Delete observations that contain a “2” in the “related” category;
    3.	*Load* the transformed data into a database;
2.	**NLP Pipeline:**
    1.	Read data from database;
    2.	Remove any categories that have only 0s or 1s as some classifiers fail for such data sets;
    3.	Tokenize the messages – remove punctuation, convert to lower case, remove stop words;
    4.	Tfidf transformation;
3.	**ML Pipeline:**
    1.	Train a RidgeClassifier and optimize its “alpha” parameter using a grid search;
    2.	Display precision, recall, f-score metrics for test data.


## Discussion<a name="discussion"></a> ##
1. Most categories appear very few times in the data set. For example, there are no occurrences of the “child_alone” category. It is difficult to train a classifier for categories (labels) where data is imbalanced, and the resulting model should be used with caution. The *class_weight='balanced'* was tested to correct for imbalance but it did not perform uniformly - it benefite some categories, but hurt other categories. Ideally, after detailed testing, the flag could be set on a category-by-category basis. This tailoring was not done in this project.
2. Trial and error showed that different classifiers (Ridge Classifier, Ada Boost, Random Forest, Decision Tree) are best-suited for different categories. Ideally, the best classifier would be chosen for each individual category. This project uses the same classifier for all categories.
3. Using the f-score as a combined performance metric, RidgeClassifier performance is not uniform across categories. 
![fscore](https://github.com/a1pat/Disaster-Pipeline/blob/main/images/fscore.jpg)
4. A plot of the **ocurrence rate** (the percentage of observations that contain a 1 for a category - a measure of imbalance) provides a first clue to the observed f-score. Categories with f-score of zero have a very low ocurrence rate. Correcting imbalance may improve fit quality as measured by the f-score (categories with ocurrence rate greater than 0.2 (20%) generally have a high f-score). 
![occurrence rate](https://github.com/a1pat/Disaster-Pipeline/blob/main/images/occurrence_rate.jpg)
5. Additional insight is provided by the metrics bubble plot:
    1. The training data is fit with high precision, but not the test data (orange bubbles are on the right-hand side of the plot, but scattered from top to bottom). The categories with low test precision also have low occurrence (small bubbles). This could be due to a combination of test set size (30% of the total data set) and imbalance.
    2. The recall is adversely affected by imbalance - the smaller green bubbles are on the lower left. Many categories have a recall rate less than 50% - this means that less than half of the requests fo rhelp will be flagged as such. **Recall is clearly an area for improvement.**
![metrics](https://github.com/a1pat/Disaster-Pipeline/blob/main/images/metrics.jpg)


## File Descriptions<a name="file_descriptions"></a> ##
The files/folders in the project's root folder are:
*	**app** (folder)
    *	**templates** (folder)
        *	**go.html:** code to show the results of the classification;
        *	**master.html:** code for main web page;
        * **run.py:** Web app back-end. Creates charts displayed on web page; runs classifier on message;
*	**data** (folder)
    * **disaster_categories.csv:** message, original message, genre for each message;
    * **disaster_messages.csv:** message is tagged as belonging to (1/0) each of thirty six pre-defined categories;
    * **DisasterResponse.db:** Database containing cleaned messages and categories from the ETL pipeline;
    * **process_data.py:** python code for the ETL pipeline;
*	**models** (folder)
    * **classifier.pkl:** Trained classifier;
    * **train_classifier.py:** python code for the NPL and ML pipelines. Reads from DisasterResponse.db and writes the model to classifier.pkl;
*   **images** (folder)
    * **fscore.jpg**: plot of training and test set fscore by category;
    * **occurrence_rate.jpg**: plot of occurrence rate (measure of imbalance) by category;
    * **metrics.jpg**: plot of training and test metrics (fscore, precision, recall);
*	**README.md:** This document.

**Run the following commands in the project's root directory to set up your database and model:**

To run ETL pipeline that cleans data and stores in database:  
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

To run ML pipeline that trains classifier and saves:  
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

**Run the following command in the app directory to run your web app:**  
`python run.py`


## Libraries/Packages<a name="libraries_packages"></a> ##
The following python packages are used:
1.	re
2.	numpy
3.	pandas
4.	sklearn
5.	nltk
6.	json
7.	plotly
8.	flask
9.	sqlalchemy
10.	sys
11.	pickle


## Acknowledgements<a name="acknowledgements"></a> ##
[Figure Eight Inc](https://www.figure-eight.com) provided the curated data set.

[Udacity](http://www.udacity.com) provided the html files, run.py, detailed project guidance and the development environment.


## Licensing<a name="licensing"></a> ##


## Authors<a name="authors"></a> ##
1. [Udacity](http://www.udacity.com) provided the html files and run.py;
2. [Figure Eight Inc](https://www.figure-eight.com) provided disaster_messages.csv and disaster_categories.csv;
3. [Ashutosh A. Patwardhan](https://github.com/a1pat) coded process_data.py and train_classifier.py; modified run.py.

