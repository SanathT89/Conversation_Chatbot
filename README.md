# Conversation_Chatbot
This is a terminal-line-based chatbot that has a conversation with the user based on the queries generated by the user. It is intended for dialogue with the user. The accuracy of the responses is determined by the confidence score indicating how confident the answer is to the query generated.

The dialogs.txt is the text file containing the set of queries that the user may ask and the responses to the asked questions. 

The ChatbotNotebook folder contains the notebook showing the data cleaning, machine model training, and parameter tuning of the dialogs.txt data.

The TalkToMe Folder contains the executable application itself.

# The Necessary Installation of Dependencies:
* Python 3.11.4
* Jupyter Notebook (If modification of machine learning models or data is to be done)
* Natural Language Toolkit (nltk) 3.8.1
* Other library requirements are specified in the notebook and are to be installed appropriately and are not required if only the application is to be run.


# Steps in running the application:
* Install all dependencies.
* Download TalkToMe folder from the repository.
* Run the terminal in the folder containing TalktoMe.py
* Run the command:
 python TalkToMe.py

# Issues to be fixed:
- [ ] Data insufficiency leads to abrupt responses in certain cases when the user runs a query.
- [ ] Text sentiment calculation can be made better than the used approach(Using VADER SentimentIntensityAnalyser).
- [ ] Terminal execution time can be made better by either connecting this to website frontend or if used as a module for a desktop application.
- [ ] Raw Data annotation can be changed for better processing. 

