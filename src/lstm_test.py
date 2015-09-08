__author__ = 'misha'


from ocrolib.lstm import *
import pickle
from gi.overrides.GLib import get_current_time


class MyParallel(Parallel):
    """Difference is that backward function returns deltas.
    """
    # def __init__(self,*nets):
    #     self.nets = nets
    def backward(self,deltas):
        deltas = array(deltas)
        new_deltas = []
        start = 0
        for i,net in enumerate(self.nets):
            k = net.noutputs()
            new_deltas.append(np.array(net.backward(deltas[:,start:start+k])))
            start += k

        # I'm not sure we should sum the deltas from parallel layers
        res = new_deltas[0]
        for i in range(1, len(self.nets)):
            res += new_deltas[i]
        return res

print str(get_current_time())



# dataset
# xs, ys = ([], [])
# for i in xrange(50):
#     array = np.random.random(10)
#     xs.append(array)
#     a = argmax(array)
#     y = zeros((10,))
#     y[a] = 1.0
#     ys.append(y)
#
# xs = np.array(xs)
# ys = np.array(ys)
#
# plot(xs, "y.")
# plot(ys, "g")

Ni = 10
Ns = 30
No = 10
model = Stacked([LSTM(Ni, Ns), LSTM(Ns, Ns), Softmax(Ns, No)])
model.setLearningRate(0.001, 0.9)

# model.lstm = Stacked([MyParallel(LSTM(Ni, Ns), Reversed(LSTM(Ni, Ns))),
#                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
#                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
#                            Softmax(2*Ns, No)])
# model.setLearningRate(0.001)

dataset = []
for s in xrange(30):
    xs, ys = ([], [])
    for k in xrange(10):
        array = np.random.random(10)
        xs.append(array)
        a = argmax(array)
        y = zeros((10,))
        y[a] = 1.0
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)
    dataset.append((xs, ys))


sq_error_log = []
for i in xrange(500):
    sq_error = 0
    for (xs, ys) in dataset:
        pred = model.train(xs, ys)
        sq_error += sum((ys-pred)**2)
    print sq_error
    sq_error_log.append(sq_error)
    if i%5 == 0:
        figure("square error"); clf();
        plot(np.array(sq_error_log), "r")
    ginput(1,0.01);
    if i%300 == 0:
        print pred
        # plot(np.array(pred), "r")
        # show()

# # reading model from file
# with open("../models/lstm_model.1", 'r') as file:
#     model = pickle.load(file)

correct = 0
total = 100
for i in xrange(total):
    array = np.random.random(10)
    print "input: " + str(array)
    res = str(argmax(model.predict([array])))
    print "result: " + res
    ans = str(argmax(array))
    print "correct: " + ans
    if res == ans: correct += 1

print "accuracy: " + str(1.0*correct/total)
# print pr
# plot(np.array(pr), "r")
# show()

# # saving model in file
# model.dstats = None
# model.ldeltas = None
# model.deltas = None
# newpath = r'../models'
# if not os.path.exists(newpath):
#     os.makedirs(newpath)
# with open("../models/lstm_model.1", 'w') as file:
#     pickle.dump(model, file)
