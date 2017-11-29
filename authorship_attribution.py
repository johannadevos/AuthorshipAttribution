# Script written by Johanna de Vos, 29/11/2017
# Based on a Python template provided by the course instructors of Text and Multimedia Mining
# Radboud University, Nijmegen

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.data import load
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Download the 'stopwords' and 'punkt' from the Natural Language Toolkit, you can comment the next lines if already present.
#nltk.download('stopwords')
#nltk.download('punkt') # This is a sentence tokenizer

              
# Create several lists that can be used in feature engineering              
stop_words = sorted(list(set(stopwords.words('english')))) # English stopwords
tagdict = load('help/tagsets/upenn_tagset.pickle') # Penn Treebank POS tags
pos_tags = list(tagdict.keys())
alphabet = list(map(chr, range(97, 123)))


# Specify the path which contains the directory where all the data are
path = "C:/..."


# Load the dataset into memory from the filesystem
def load_data(dir_name):
    return sklearn.datasets.load_files(path + '/%s' % dir_name, encoding='utf-8')


# Load the training data
def load_train_data():
    return load_data('train')


# Load the testing data
def load_test_data():
    return load_data('test')


# Extract features from a given text
def extract_features(text):
    
    bag_of_words = [x for x in wordpunct_tokenize(text)] # List of all words
    bag_without_stops = [x for x in wordpunct_tokenize(text) if x.lower() not in stop_words] # List of words without stops
    sentences = [x for x in sent_tokenize(text)] # List of sentences
    sen_length = [len(x) for x in sentences] # List of sentence lengths
    BOW_sen = [wordpunct_tokenize(x) for x in sentences] # List of bag of words for each sentence
    
    feat_values = []
    feat_names = []
    feat_groups = []
    
    
    # GROUP A: length features

    feat_group = 'A'
    
    feat_values.append(len(bag_of_words))
    feat_names.append('Total number of words (stopwords included)')
    feat_groups.append(feat_group)
 
    feat_values.append(len(bag_without_stops))
    feat_names.append('Total number of words (stopwords excluded)')
    feat_groups.append(feat_group)

    feat_values.append(len(sentences))
    feat_names.append('Number of sentences')
    feat_groups.append(feat_group)

    feat_values.append(sum(sen_length) / len(sen_length))
    feat_names.append('Average sentence length in characters')
    feat_groups.append(feat_group)
    
    feat_values.append(sum([len(x) for x in BOW_sen]) / len(BOW_sen))
    feat_names.append('Average sentence length in words')
    feat_groups.append(feat_group)
    
            
    # GROUP B: lexical richness
    
    feat_group = 'B'
    
    feat_values.append(len(bag_without_stops) / len(bag_of_words))
    feat_names.append('Ratio of lexical words to total number of words')
    feat_groups.append(feat_group)

    feat_values.append(len(set(bag_of_words)) / len(bag_of_words))
    feat_names.append('Type-token ratio (stopwords included)')
    feat_groups.append(feat_group)

    feat_values.append(len(set(bag_without_stops)) / len(bag_without_stops))
    feat_names.append('Type-token ratio (stopwords excluded)')
    feat_groups.append(feat_group)

    feat_values.append(len(set(bag_of_words)))
    feat_names.append('Number of different words (stopwords included)')
    feat_groups.append(feat_group)

    feat_values.append(len(set(bag_without_stops)))
    feat_names.append('Number of different words (stopwords excluded)')    
    feat_groups.append(feat_group)
        
    feat_values.append(len(set(bag_of_words[:50])))    
    feat_names.append('Number of different words in first 50 words (stopwords included)')
    feat_groups.append(feat_group)
    
    feat_values.append(len(set(bag_without_stops[:50])))    
    feat_names.append('Number of different words in first 50 words (stopwords excluded)')
    feat_groups.append(feat_group)
   
    
    # GROUP C: frequency of function words
    
    feat_group = 'C'
    
    for x in range(len(stop_words)):
        feat_values.append(bag_of_words.count(stop_words[x]))
        feat_names.append('Frequency of stopword: %s' % stop_words[x])
        feat_groups.append(feat_group)
    

    # GROUP D: frequencies of part of speech (BOW level)
    
    feat_group = 'D'
    
    tags = pos_tag(bag_of_words)
    list_of_tags = [tuple[1] for tuple in tags]
    
    for x in range(len(pos_tags)):
        feat_values.append(list_of_tags.count(pos_tags[x]))
        feat_names.append('Frequency of POS tag: %s' % pos_tags[x])   
        feat_groups.append(feat_group)
    
    
    # GROUP E: character frequencies
    
    feat_group = 'E'
    
    for x in range(len(alphabet)):
        feat_values.append(text.lower().count(alphabet[x]))
        feat_names.append('Frequency of character: %s' % alphabet[x])
        feat_groups.append(feat_group)

    
    return feat_values, feat_names, feat_groups
    

# Train classifier on the training set, and predict the validation set
def classify(train_features, train_labels, test_features):
    
    # Optional: If you would like to test different how classifiers would perform different, you can alter the classifier here.
    
    clf = SVC(kernel='linear')
    clf.fit(train_features, train_labels)
    
    return clf.predict(test_features)


# Evaluate predictions (y_pred) given the ground truth (y_true)
def evaluate(y_true, y_pred):

    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("F1-score (harmonic mean): %f" % f1_score)
    
    return recall, precision, f1_score

    
# Classify and evaluate using k-fold cross-validation
def cross_validate(train_data, feat_values):
    
    skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
    scores = []
    for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(train_data.filenames, train_data.target)):
        
        # Print the fold number
        print("Fold %d" % (fold_id + 1))
        
        # Collect the data for this train/validation split
        train_features = [feat_values[x] for x in train_indexes]
        train_labels = [train_data.target[x] for x in train_indexes]
        validation_features = [feat_values[x] for x in validation_indexes]
        validation_labels = [train_data.target[x] for x in validation_indexes]

        # Classify and add the scores to be able to average later
        y_pred = classify(train_features, train_labels, validation_features)
        scores.append(evaluate(validation_labels, y_pred))

        # Print a newline
        print("")
    
    # Print the averaged score
    recall = sum([x[0] for x in scores]) / len(scores)
    print("Averaged total recall", recall)
    precision = sum([x[1] for x in scores]) / len(scores)
    print("Averaged total precision", precision)
    f_score = sum([x[2] for x in scores]) / len(scores)
    print("Averaged total f-score", f_score)
    print("")
    
    return recall, precision, f_score
    
    
# Leave out one feature group at a time
def leave_group_out(train_data, test_data, feat_values_tr, feat_values_te, feat_names, feat_groups):
    
    group_letters = sorted(set(feat_groups))
    results = []
    
    # Get a list of indices for the start of each new group
    group_indices = [feat_groups.index(group) for group in group_letters]

    # Classify and evaluate, leaving one feature group out at a time
    counter_tr = 0
    counter_te = 0
    
    for group in group_letters:
        print(group)
        index_start = group_indices[counter_tr]
        
        if group is not group_letters[-1]:
            index_end = group_indices[counter_tr + 1]
            feat_values_min_group_tr = [x[:index_start] + x[index_end:] for x in feat_values_tr]
        elif group is group_letters[-1]:
            feat_values_min_group_tr = [x[:index_start] for x in feat_values_tr]
        
        counter_tr += 1
        
        # For the training data, cross-validate
        if not test_data:
            recall, precision, f_score = cross_validate(train_data, feat_values_min_group_tr)        
            results.append((group, recall, precision, f_score))
        
        # For the test data, only classify and evaluate the test set
        if test_data:
            print ("test")
            
            if group is not group_letters[-1]:
                index_end = group_indices[counter_te + 1]
                feat_values_min_group_te = [x[:index_start] + x[index_end:] for x in feat_values_te]
            elif group is group_letters[-1]:
                feat_values_min_group_te = [x[:index_start] for x in feat_values_te]
                
            y_pred = classify(feat_values_min_group_tr, train_data.target, feat_values_min_group_te)
            recall, precision, f_score = evaluate(test_data.target, y_pred)
            results.append((group, recall, precision, f_score))
            
            counter_te += 1
    
    return results

   
# Create a bar plots of the results
def bar_plot(group_results):
    
    # Create pandas dataframe
    df = pd.DataFrame(group_results)
    df.columns = ['Feature group', 'Recall', 'Precision', 'F1 score']
        
    # Data visualisation: recall, precision and F1
    df.plot.bar(x = 'Feature group', legend = True, ylim = (0,1), title = 'Failure analysis')
    
    # F1 only
    df.plot.bar(x = 'Feature group', y = 'F1 score', legend = True, ylim = (0,0.75), title = 'Failure analysis')
    
   
# Create and plot the confusion matrix
def conf_matrix(y_true, y_pred):
    
    # Create a list of (actual, predicted) pairs
    pairs = list(zip(y_true, y_pred))
    
    # Create and fill confusion matrix
    conf_m = np.zeros(shape=(45,45))
    
    for x in range(len(y_true)):
        conf_m[pairs[x][0], pairs[x][1]] += 1
         
    # Plot confusion matrix
    plt.matshow(conf_m)
    plt.title('Confusion matrix\n')
    plt.xlabel(s = 'Predicted author', verticalalignment = 'top')
    plt.ylabel('Actual author')
    plt.colorbar()
    plt.show()
    
    # Write array to file
    #np.savetxt(fname = 'Confusion matrix.txt', X = conf_m, delimiter = " ")

    
# The main program
def main():
    train_data = load_train_data()

    # TRAINING AND VALIDATION SET
    
    # Extract the features
    print("Extracting features...")
    feat_values_tr, feat_names, feat_groups = list(zip(*map(extract_features, train_data.data)))
    feat_names = feat_names[0]
    feat_groups = feat_groups[0]
  
    # Classify and evaluate using k-fold cross-validation
    recall_tr, precision_tr, f_score_tr = cross_validate(train_data, feat_values_tr)

    # Leave out one feature group at a time
    group_results_tr = leave_group_out(train_data, None, feat_values_tr, None, feat_names, feat_groups)
    
    # Bar plots
    group_results_tr.insert(0, ('None', recall_tr, precision_tr, f_score_tr))
    bar_plot(group_results_tr)        

    # TEST SET
    
    # Extract the features
    test_data = load_test_data()
    feat_values_te, feat_names, feat_groups = list(zip(*map(extract_features, test_data.data)))
    feat_names = feat_names[0]
    feat_groups = feat_groups[0]
    
    # Classify and evaluate
    y_pred = classify(feat_values_tr, train_data.target, feat_values_te)
    recall_te, precision_te, f_score_te = evaluate(test_data.target, y_pred)
    
    # Leave out one feature group at a time
    group_results_test = leave_group_out(train_data, test_data, feat_values_tr, feat_values_te, feat_names, feat_groups)
    
    # Visualise (test data)
    group_results_test.insert(0, ('None', recall_te, precision_te, f_score_te))
    bar_plot(group_results_test) 
    conf_matrix(test_data.target, y_pred)       
    

# This piece of code is common practice in Python, it is something like if "this file" is the main file to be ran, then execute this remaining piece of code. The advantage of this is that your main loop will not be executed when you import certain functions in this file in another file, which is useful in larger projects.
if __name__ == '__main__':
    main()