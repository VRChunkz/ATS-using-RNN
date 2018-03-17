from pybrain.structure import RecurrentNetwork
from pybrain.structure import LSTMLayer, LinearLayer, SoftmaxLayer
from pybrain.structure import FullConnection
from vectorizer_engine import VectorizerEngine

# Initialize vector engine
vec_engine = VectorizerEngine()

# Initialize a recurrent network
# Input layer will be linear layer of dimension equal to our vectorizer_engine's dimension
# Output layer will be a softmax layer of dimension equal to our vectorizer_engine's dimension
# Hidden layer is LSTM layer given a dimension of an arbitary number, say 5.
net = RecurrentNetwork()
in_layer = LinearLayer(vec_engine.word_vec_dim, name="input_layer")
hidden_layer = LSTMLayer(5, name="hidden_layer")
out_layer = SoftmaxLayer(vec_engine.word_vec_dim, name="out_layer")

# Connecting between layers. And a special connection from out to hidden, that is the recurrent connection
conn_in_to_hid = FullConnection(in_layer, hidden_layer, name="in_to_hidden")
conn_hid_to_out = FullConnection(hidden_layer, out_layer, name="hidden_to_out")
recurrent_connection = FullConnection(hidden_layer, hidden_layer, name="recurrent")

# Putting everything together.
net.addInputModule(in_layer)
net.addModule(hidden_layer)
net.addOutputModule(out_layer)

net.addConnection(conn_in_to_hid)
net.addConnection(conn_hid_to_out)
net.addRecurrentConnection(recurrent_connection)

net.sortModules()

# Since our preprocessor_engine does stuff and writes output to output.txt,
# neural_engine takes its input from it
input_file = open('output.txt', 'r')

# each line in output.txt is a preprocessed token.
# Read line by line and remove endline character
input_tokens = input_file.readlines()
input_tokens = [t.strip() for t in input_tokens]
input_file.close()

# for each token, convert it to vector of dimension equal to our vectorizer_engine's dimension
# send that vector to neural network
# output of the network at every time step will be a vector of dimension equal to our vectorizer_engine's dimension
# convert that vector to word using our vectorizer engine and write it to file 'summary-output.txt'
summary_file = open('summary-text.txt', 'w+')
for word in input_tokens:
    input_vec = vec_engine.word2vec(word)
    output_vec = net.activate(input_vec)
    output_word = vec_engine.vec2word(output_vec)
    summary_file.write(output_word + " ")

summary_file.flush()
summary_file.close()
