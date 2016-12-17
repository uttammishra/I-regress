# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# shuffle
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

# logging
import logging
import os.path
import sys
import _pickle as pickle
import csv

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

#read = csv.reader(open("input.csv"))
#write_informative = open("write_informative.txt","w")
#write_persuasive = open("write_persuasive.txt","w")
#write_transformative = open("write_transformative.txt","w")
#one = '1'

#for row in read:
#    temp = str(row).split(',')
#    if(temp[1].find("1") == 2):
#        write_informative.write(temp[0].lstrip("[")+"\n")
#    if(temp[2].find("1") == 2):
#        write_persuasive.write(temp[0].lstrip("[")+"\n")
#    if(temp[3].find("1") == 2):
#        write_transformative.write(temp[0].lstrip("[")+"\n")

#write_informative.close()
#write_persuasive.close()
#write_transformative.close()

sources = {'write_informative.txt':'INFORMATIVE', 'write_persuasive.txt':'PERSUASIVE', 'write_transformative.txt':'TRANSFORMATIVE' ,
           'write_informative_test.txt':'INFORMATIVE_TEST', 'write_persuasive_test.txt':'PERSUASIVE_TEST', 'write_transformative_test.txt':'TRANSFORMATIVE_TEST'}

sentences = LabeledLineSentence(sources)

#model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

#model.build_vocab(sentences.to_array())

#for epoch in range(50):
#    logger.info('Epoch %d' % epoch)
#    model.train(sentences.sentences_perm())

#model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')

logger.info('Sentiment')
train_arrays = numpy.zeros((1100, 100))
train_labels = numpy.zeros(1100)

for i in range(300):
    if(i < 300):
        prefix_train_inf = 'INFORMATIVE_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_inf]
        train_labels[i] = 0
for i in range(400):
    if(i < 400):
        prefix_train_per = 'PERSUASIVE_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_per]
        train_labels[i] = 1
for i in range(400):
    if(i < 400):
        prefix_train_tra = 'TRANSFORMATIVE_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_tra]
        train_labels[i] = 2

print(train_labels)

test_arrays = numpy.zeros((1100, 100))
test_labels = numpy.zeros(1100)

for i in range(149):
    if(i < 149):
        prefix_train_inf = 'INFORMATIVE_TEST_' + str(i)
        test_arrays[i] = model.docvecs[prefix_train_inf]
        test_labels[i] = 0
for i in range(184):
    if(i < 184):
        prefix_train_per = 'PERSUASIVE_TEST_' + str(i)
        test_arrays[i] = model.docvecs[prefix_train_per]
        test_labels[i] = 1
for i in range(157):
    if(i < 157):
        prefix_train_tra = 'TRANSFORMATIVE_TEST_' + str(i)
        test_arrays[i] = model.docvecs[prefix_train_tra]
        test_labels[i] = 2

print(test_labels)

logger.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print(classifier.score(test_arrays, test_labels))

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
test_labels = numpy.array([number[0] for number in lb.fit_transform(test_labels)])

precision = cross_validation.cross_val_score(classifier, train_arrays, test_labels, cv=10, scoring='precision')
print('Precision', numpy.mean(precision))

recall = cross_validation.cross_val_score(classifier, train_arrays, test_labels, cv=10, scoring='recall')
print('recall', numpy.mean(recall))

F1 = cross_validation.cross_val_score(classifier, train_arrays, test_labels, cv=10, scoring='f1')
print('F1', numpy.mean(F1))
