import pandas as pd
import numpy as np
import json
from newspaper import Article
from rake_nltk import Rake
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups




file = open("example_1.json").read()
arr = json.loads(file)
count = 0
all_documents = []
for article in arr:
    if "content" in article and article['source_content'] != 'video' and article['validated'] == -2:

        url = article['_id']
        art = Article(url, language='en')  # English
        try:
            art.download()
            art.parse()
            art_content =  art.text
            #print(article['_id'])
            all_documents.append(art_content)
            count = count + 1
            #print(count)
            if(count == 93):
                break
        except:
            print('bad article')
            print(article['source_content'])
            print(article['_id'])
            continue

print(type(all_documents))

test_d =  pd.read_csv("/Users/eowyna/Documents/GitHub/UserPref/Test.tsv", sep='\t', encoding='latin-1')

test_data = test_d.iloc[:, 0].tolist()

print(type(test_data))



#categories = ["comp.graphics","sci.space","rec.sport.baseball"]
#newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
#newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

#print('total texts in train:',len(newsgroups_train.data))
#print('total texts in test:',len(newsgroups_test.data))

#print('text',newsgroups_train.data[0])
#print('category:',newsgroups_train.target[0])

list_labels = [0] * 93
testlist_labels = [0]*20
list_labels[2] = 1
list_labels[6] = 1
list_labels[12] = 1
list_labels[14] = 1
list_labels[38] = 1
list_labels[42] = 1
list_labels[47] = 1
list_labels[49] = 1
list_labels[50] = 1
list_labels[52] = 1
list_labels[54] = 1
list_labels[56] = 1
list_labels[60] = 1
list_labels[61] = 1
list_labels[65] = 1
list_labels[68] = 1
list_labels[73] = 1
list_labels[78] = 1
list_labels[85] = 1
list_labels[90] = 1
testlist_labels[0] = 1
testlist_labels[1] = 1
testlist_labels[2] = 1
testlist_labels[3] = 1
testlist_labels[4] = 1



vocab = Counter()

for text in range(len(all_documents)):
    content = all_documents[text]
    for word in content.split(' '):
        vocab[word.lower()]+=1

for text in range(len(test_data)):
    content1 = test_data[text]
    for word in content1.split(' '):
        vocab[word.lower()]+=1



print("Total words:",len(vocab))

total_words = len(vocab)

def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index

word2index = get_word_2_index(vocab)


print("Index of the word 'is':",word2index['is'])


def get_batch(data,label,i,batch_size):
    batches = []
    results = []
    texts = data[i*batch_size:i*batch_size+batch_size]
    categories = label[i*batch_size:i*batch_size+batch_size]

    for i in range(len(texts)):

        text = texts[i]
        layer = np.zeros(total_words,dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1
        batches.append(layer)

    for category in categories:
        y = np.zeros((2),dtype=float)
        if category == 0:
            y[0] = 1.
        else:
            y[1] = 1.
        results.append(y)


    return np.array(batches),np.array(results)


learning_rate = 0.01
training_epochs = 10
batch_size = 15
display_step = 1

n_hidden_1 = 100      # 1st layer number of features
n_hidden_2 = 100       # 2nd layer number of features
n_input = total_words # Words in vocab
n_classes = 2

input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")
output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output")


def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)

    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)

    # Output layer
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition


    # Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
prediction = multilayer_perceptron(input_tensor, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()



# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(all_documents)/batch_size)

        # Loop over all batches
        for i in range(total_batch):

            batch_x,batch_y = get_batch(all_documents,list_labels,i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")


    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(test_data)
    batch_x_test,batch_y_test = get_batch(test_data, testlist_labels,0,total_test_data)
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))
