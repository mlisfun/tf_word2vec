import urllib.request
import collections
import math
import os
import random
import zipfile
import datetime as dt

import numpy as np
import tensorflow as tf

BASE_DIR = 'D:/Python/myprojects/ml/bible/'
training_file_name = BASE_DIR + 'BibleNIV.txt'
#training_file_name = BASE_DIR + 'test.txt'
WINDOW_SIZE = 3
EMBEDDING_DIM = 100
punctuations = ['?','.',',','\"','\'', '(',')',';',':','!' ]
unused_words = ['is','a','the','in','of','on','and','as','or']
vocabulary_size = 10000

def array_sum(ar):
    ttl = 0
    print(' shape of array ',ar.shape)
    for first in range(ar.shape[1]):
        for element in ar[:,first]:
            ttl += element
        print( first, ' ttl = ',ttl)
    return ttl

def onehot_index (array):
    idx = 0
    for element in array:
        idx += i
        if element == 1:
            break            
    return idx


def load_test_data():
    corpus_raw = 'He is the king . The king is royal . She is the royal  queen . she is also wife of king'
    corpus_raw = 'And there was evening , and there was morning-the third day . And God said , Let there be lights in the expanse of the sky to separate the day from the night, and let them serve as signs to mark seasons and days and years and let them be lights in the expanse of the sky to give light on the earth . And it was so . '


    # convert to lower case
    corpus_raw = corpus_raw.lower()
    

    words = []
    for word in corpus_raw.split():
        if word != '.': # because we don't want to treat . as a word
            words.append(word)

    words = set(words) # so that all duplicate words are removed
    word2int = {}
    int2word = {}
    vocab_size = len(words) # gives the total number of unique words

    for i,word in enumerate(words):
        word2int[word] = i
        int2word[i] = word

    # raw sentences is a list of sentences.
    raw_sentences = corpus_raw.split('.')
    sentences = []
    for sentence in raw_sentences:
        sentences.append(sentence.split())


    data = []
    for sentence in sentences:
        
        for word_index, word in enumerate(sentence):
            for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
                if nb_word != word:
                    data.append([word, nb_word])

    print('size of data', len(data))
    print(data)
    x_train = [] # input word
    y_train = [] # output word

    for data_word in data:
        x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
        y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

    # convert them to numpy arrays
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train, y_train


    

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp


def write_data(filename, array_data, write_mode='w'):
    """ Write array data to list."""
    print('writing to file ', filename)
    with open(filename,write_mode) as f:
        f.write(str(array_data))    

def preprocess_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    data = []
    print('processing file ', filename)
    with open(filename) as f:
        for line in f.readlines():
            text = line
            for item in text.split():
                if len(item) >0 :
                    if item[0].isdigit():
                        item = ''.join(i for i in item if not i.isdigit())
                        
                    for rep in punctuations:
#                        log_str = 'rep -' + rep  + '  - '+ item + ' -> ' 
                        item = item.replace(rep,'')
#                        print( log_str ,  item) 
                        
                    if len(item) >0 :
                        item = item.lower()
                        if item not in unused_words:
                            data.append(item)
#            quit()

        f.close()
    return data

## ##################################
##  --->>>    building  onehot vectors for training
## ##################################
## def generate_batch(data, batch_size, num_skips, skip_window):
def get_training_data(filename, line_start, batch_size):
    """ buliding onehot vector for training purpose."""
    data = []
    onehot_input = []
    onehot_label = []
    line_index = 0
    batch_ended = False
#    print('processing file ', filename)
    with open(filename) as f:
        for line in f.readlines():
            line_index +=1
            if line_index> line_start:
                sentences = line
#                print('sentences ', sentences)
                for sentence in sentences.split('.'):
                    words = []  # reset words arrays
#                    print(line_index,  ' sentence ', sentence)
                    for item in sentence.split():
                        if len(item) >0 :
                            if item[0].isdigit():
                                item = ''.join(i for i in item if not i.isdigit())
                                
                            for rep in punctuations:
                                log_str = 'rep -' + rep  + '  - '+ item + ' -> ' 
                                item = item.replace(rep,'')
    #                            print( log_str ,  item) 
                                
                        if len(item) >0 :
                            item = item.lower()
                            if item in dictionary:
#                            item not in unused_words:
                                words.append(item)
                        
                                
  #                  print('words ', words)
                    ### put processed words into array
                    for word_index, word in enumerate(words):
  #                      print('word_index',word_index, word, max(word_index - WINDOW_SIZE, 0) , min(word_index + WINDOW_SIZE, len(sentence)) + 1)
                        for nb_word in words[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
                            if nb_word != word:
                                if len(onehot_input) < batch_size:
#                                    oneshot_data.append([word, nb_word])
                                    data_point_index = dictionary[word]
  #                                  onehot_input.append(data_point_index)
                                    in_onehot = to_one_hot(data_point_index, vocabulary_size)
                                    data_point_index = dictionary[nb_word]                                    
#                                    onehot_label.append(data_point_index)
                                    lb_onehot = to_one_hot(data_point_index, vocabulary_size)
                                    onehot_input.append(in_onehot)
                                    onehot_label.append( lb_onehot)

                                    
                                else:
                                    batch_ended = True
                                    break
                        if batch_ended :
                            break
                    if batch_ended:
                        break
                if batch_ended:
                    break
            if batch_ended:
                break
                        

        f.close()
#    print('onehot_input ', in_onehot)
 #   print('lb_input ', lb_onehot)
    

    onehot_input= np.asarray(onehot_input)
    onehot_label= np.asarray(onehot_label)
    return onehot_input, onehot_label, line_index



def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def collect_data(vocabulary_size=10000):
#    filename = BASE_DIR + 'text8.txt'
    filename = training_file_name
    vocabulary = preprocess_data(filename)
## >>>>>>>
#    quit()
    
    print('vocabulary  size  ', len(vocabulary),  vocabulary[:5])
    write_data(BASE_DIR+'vocabulary.txt',vocabulary)
    
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary

data_index = 0


#    """ data = array of numbers,  batch_size = 128, num_skips = 2, skip_window = 2 ."""
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    print('Generate batch start ', data_index, batch_size, skip_window)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    print('Generate batch end ', data_index)
    return batch, context




data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocabulary_size)

data_file_name = BASE_DIR+'count.txt'
write_data(data_file_name, count )
data_file_name = BASE_DIR+'dictionary.txt'
write_data(data_file_name, dictionary)
data_file_name = BASE_DIR+'reverse_dictionary.txt'
write_data(data_file_name, reverse_dictionary)

line_index = 0
#training_data_oneshot,lline_index = get_training_data(training_file_name, line_index)

batch_size = 128
# embedding_dim = 300  # Dimension of the embedding vector. refer to EMBEDDING_DIM
skip_window  = WINDOW_SIZE       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 6     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
validate_words = {'lord','peter','you','grace','pray','david','jesus'}
#validate_examples = np.random.choice(valid_window, valid_size, replace=False)
validate_examples = []
for v_word in validate_words:
    idx = dictionary[v_word]
    onehot = to_one_hot(idx,vocabulary_size)
#    validate_examples.append(onehot)
    validate_examples.append(idx)
    



num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():


    # making placeholders for x_train and y_train
    train_inputs = tf.placeholder(tf.float32, shape=(None, vocabulary_size))
    train_context = tf.placeholder(tf.float32, shape=(None, vocabulary_size))

#    EMBEDDING_DIM = 5 # you can choose your own number


    W1 = tf.Variable(tf.random_normal([vocabulary_size, EMBEDDING_DIM]))
    b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
    hidden_representation = tf.add(tf.matmul(train_inputs,W1), b1)
    W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocabulary_size]))
    b2 = tf.Variable(tf.random_normal([vocabulary_size]))
    prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))



    # define the loss function:
    cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(train_context * tf.log(prediction), reduction_indices=[1]))

    # define the training step:
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
#    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(prediction)

###########################
# codes for finding nearest neighbours
###########################

    validate_dataset = tf.constant(validate_examples, dtype=tf.int32)
  
 # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0))
#NOT USED    embed = tf.nn.embedding_lookup(embeddings, train_inputs)



  # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm  ## embedding = array of [vacab size, embbeding_size =300  ]
    validate_embeddings = tf.nn.embedding_lookup(normalized_embeddings, validate_dataset)   ## valid_dataset = valid_examples[16] single dimension array
    similarity = tf.matmul(validate_embeddings, normalized_embeddings, transpose_b=True)





##  # Look up embeddings for inputs.
##    test_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
##    embed = tf.nn.embedding_lookup(test_embeddings, train_inputs)
##    print('embed ', type(embed),  embed.shape)
#    sess = tf.Session()
#    init = tf.global_variables_initializer()
 #   sess.run(init) #make sure you do this!

def run(graph, num_steps):
    data_index = 0
    data_index_start = 0
    with tf.Session(graph=graph) as session:
        init = tf.global_variables_initializer()
        session.run(init) #make sure you do this!
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_context, data_index_start  = get_training_data(training_file_name, data_index_start ,batch_size)
#            print('batch_inputs ', len(batch_inputs) , batch_inputs.shape, len(batch_context), batch_context.shape)
 #           write_data(BASE_DIR+'batch_input.txt', batch_inputs)
#            print('batch_inputs ', array_sum(batch_inputs))
#            print('batch_context ', array_sum(batch_context))


            
            feed_dict = {train_inputs: batch_inputs, train_context: batch_context}
            n_iters = 5
            for i in range(n_iters):
                test = session.run(train_step, feed_dict=feed_dict)
#                weight1 = session.run(W1)
#                write_data(BASE_DIR+'weight1.txt', weight1)
 #               weight2 = session.run(W2)
 #               print(step ,i, '  weights W ', array_sum(weight1), array_sum(weight2))
#                write_data(BASE_DIR+'weight2.txt', weight2)

            

#            print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
            if step % 50 == 0:
                print(step, ' - loss is : ', session.run(cross_entropy_loss, feed_dict=feed_dict))


            # finding nearest words
            if step % 100 == 0:
                sim = similarity.eval()
                i = 0
                for word in validate_words:
#                    i = dictionary[word]
                    print(' validating word' , word,  i)
                    print( ' sim shape ', sim.shape)
                    top_nearest = 12
                    nearest  = (-sim[i,:]).argsort()[1:top_nearest +1]
                    log_str = 'Nesrest to %d %s:' %(i, word)
                    for k in range(top_nearest):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s %s,'% ( log_str,str(nearest[k]), close_word)
                    print(log_str)
                    i +=1

            final_embeddings = normalized_embeddings.eval()
            

      

        

num_steps = 1000
softmax_start_time = dt.datetime.now()
run(graph, num_steps=num_steps)
softmax_end_time = dt.datetime.now()
print("Softmax method took {} minutes to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))


