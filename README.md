# Natural-Language-Inference-SNLI
This is a three class classification problem in which two sentences are given known as premise and hypothesis. Three labels entailed, neutral, and contradiction are 
assigned based on whether hypothesis can be inferred from the given premise or not.
Two classifiers are trained from scratch.
One is logistic regression classifier trained using TF-IDF features generated using WordNet Lemmatizer.
Second is Bidirection LSTM classifier trained using GloVe 6B word vectors.
Classfication accuracy of 64.44% is achieved in logistic regression classifier and 80.85% is achieved in Bidirectional LSTM classifier.
