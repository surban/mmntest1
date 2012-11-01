import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
import svm, svmutil

from mlp import MLP
from logistic_sgd import LogisticRegression, load_data

dataset='mnist.pkl.gz'
batch_size=20
learning_rate=0.01
L1_reg=0.00
L2_reg=0.0001
n_epochs=1000
do_svm_training = False

def sv_to_vec(sv, length):
    vec = numpy.zeros(length)
    for indx, val in sv.iteritems():
        if indx != -1:
            vec[indx-1] = val
    return vec

def get_svm_weights(m, n_inputs):
    svcoef = m.get_sv_coef()
    svs = m.get_SV()
    fullsvs = [sv_to_vec(sv, n_inputs) for sv in svs]
    w = numpy.zeros(n_inputs)
    for i in range(len(fullsvs)):
        w += svcoef[i][0] * fullsvs[i]
    b = -m.rho[0]

    if m.label[0] == -1:
        w = -w
        b = -b
    if m.nr_class == 1:
        if m.label[0] == 1:
            b = 1
        else:
            b = -1
    return w, b

if __name__ == "__main__":
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    #trsx = train_set_x.get_value()
    #trsy = train_set_y.get_value()
    #train_set_y = T.cast(train_set_y, 'int32')
    valid_set_x, valid_set_y = datasets[1]
    vasx = valid_set_x.get_value()
    vasy = valid_set_y.get_value()
    valid_set_y = T.cast(valid_set_y, 'int32')
    test_set_x, test_set_y = datasets[2]
    tesx = test_set_x.get_value()
    tesy = test_set_y.get_value()
    test_set_y = T.cast(test_set_y, 'int32')

    print "# training batches %d" % (train_set_x.get_value(borrow=True).shape[0] / batch_size)

    # make trainset smaller
    #tsize = 200
    tsize = train_set_x.get_value(borrow=True).shape[0] / batch_size
    train_set_x.set_value(train_set_x.get_value()[0:tsize*batch_size,:])
    train_set_y.set_value(train_set_y.get_value()[0:tsize*batch_size])
    trsx = train_set_x.get_value()
    trsy = train_set_y.get_value()
    train_set_y = T.cast(train_set_y, 'int32')

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=500, n_out=10)

    # load trained parameters
    params = numpy.load("mlp_mnist.npz")
    classifier.hiddenLayer.W.set_value(params['hidden_W'])
    classifier.hiddenLayer.b.set_value(params['hidden_b'])
    classifier.logRegressionLayer.W.set_value(params['logreg_W'])
    classifier.logRegressionLayer.b.set_value(params['logreg_b'])

    # test model functions
    train_loss = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # calculate current loss
    train_losses = [train_loss(i) for i in xrange(n_train_batches)]
    train_score = numpy.mean(train_losses)
    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
    validation_score = numpy.mean(validation_losses)
    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_score = numpy.mean(test_losses)
    print "BP: Training: %.3f%%, Validation: %.3f%%, Test: %.3f%%" % \
        (train_score * 100., validation_score * 100., test_score * 100.)

if __name__ == "__main__" and do_svm_training:
    # calculate targets hl_targets[sample, hidden_neuron] for hidden layer
    print "Calculating SVM targets..."
    hl = classifier.hiddenLayer
    hl_out = theano.function(inputs=[index],
                             outputs=hl.output,
                             givens={x: train_set_x[index * batch_size:
                                                    (index + 1) * batch_size]})
    hl_targets = numpy.zeros((n_train_batches*batch_size, hl.n_out))
    for b in range(n_train_batches):
        hl_targets[b*batch_size : (b+1)*batch_size, :] = numpy.sign(hl_out(b))

    # svm_targets[neuron][sample]
    svm_targets = [list(hl_targets[:, n]) for n in range(hl.n_out)]
    #svm_target_array = numpy.asarray(svm_targets).T
    # svm_inputs[sample][input]
    tsx = train_set_x.get_value()
    svm_inputs = [list(tsx[s, :]) for s in range(tsx.shape[0])]

    # use targets to train one svm for each hidden neuron
    print "Training SVMs..."
    probs = []
    params = []
    svms = []
    ws = []
    bs = []
    werrs = 0
    for n in range(hl.n_out):
        print "Hidden neuron: %d" % n,
        print " Problem...",
        if n == 0:
            probs.append(svmutil.svm_problem(svm_targets[n], svm_inputs))
        else:
            probs.append(svmutil.svm_problem(svm_targets[n], None, tmpl=probs[0]))
        params.append(svmutil.svm_parameter("-q -s 0 -t 0 -c 100"))
        print " Training...",
        svms.append(svmutil.svm_train(probs[n], params[n]))
        print " Saving...",
        svmutil.svm_save_model("hidden%04d.svm" % n, svms[n])

        print " Testing..."
        # get weights from SVM
        w, b = get_svm_weights(svms[n], hl.n_in)
        ws.append(w)
        bs.append(b)

        # test model
        predv = numpy.dot(w, trsx.T) + b
        pred = numpy.sign(predv)
        pos = 0
        neg = 0        
        for i in range(pred.size):
            if svm_targets[n][i] > 0:
                pos += 1
            else:
                neg += 1
            if pred[i] != svm_targets[n][i]:
                print "%d: is: %f, should: %f" % (i, pred[i], svm_targets[n][i])
                print "%d: prediction value: %f" % (i, predv[i])
                werrs += 1                    
        #print "Neuron %d: #positive: %d, #negative: %d" % (n, pos, neg)
        #if werrs > 1:
        #    raise Exception()

    print "Done, hidden layer prediction errors: %d" % werrs

    # construct weight and bias matrix
    svm_hidden_W = numpy.asarray(ws).T
    svm_hidden_b = numpy.asarray(bs)
    svm_logreg_W = classifier.logRegressionLayer.W.get_value()
    svm_logreg_b = classifier.logRegressionLayer.b.get_value()

    # save SVM weights and biases
    print "Saving..."
    numpy.savez_compressed("svm1_mnist.npz", 
                           hidden_W=svm_hidden_W,
                           hidden_b=svm_hidden_b,
                           logreg_W=svm_logreg_W,
                           logreg_b=svm_logreg_b)

if __name__ == "__main__" and not do_svm_training:
    #print "Loading..."
    params = numpy.load("svm1_mnist.npz")
    svm_hidden_W = params['hidden_W']
    svm_hidden_b = params['hidden_b']
    svm_logreg_W = params['logreg_W']
    svm_logreg_b = params['logreg_b']

if __name__ == "__main__":
    # construct model using SVMs
    svm_classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=500, n_out=10)
    svm_classifier.logRegressionLayer.W.set_value(svm_logreg_W)
    svm_classifier.logRegressionLayer.b.set_value(svm_logreg_b)
    svm_classifier.hiddenLayer.W.set_value(svm_hidden_W)
    svm_classifier.hiddenLayer.b.set_value(svm_hidden_b)

    # test model functions
    svm_train_loss = theano.function(inputs=[index],
            outputs=svm_classifier.errors(y),
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    svm_test_model = theano.function(inputs=[index],
            outputs=svm_classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    svm_validate_model = theano.function(inputs=[index],
            outputs=svm_classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # calculate current loss
    svm_train_losses = [svm_train_loss(i) for i in xrange(n_train_batches)]
    svm_train_score = numpy.mean(svm_train_losses)
    svm_validation_losses = [svm_validate_model(i) for i in xrange(n_valid_batches)]
    svm_validation_score = numpy.mean(validation_losses)
    svm_test_losses = [svm_test_model(i) for i in xrange(n_test_batches)]
    svm_test_score = numpy.mean(svm_test_losses)
    print "SVM: Training: %.3f%%, Validation: %.3f%%, Test: %.3f%%" % \
        (svm_train_score * 100., svm_validation_score * 100., svm_test_score * 100.)

    print 'Training SVM logreg layer...'

    # train logistic regression layer
    cost = svm_classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr

    # update only logreg parameters using gradient descent
    svm_updates = {param: param - learning_rate * T.grad(cost, param) 
                   for param in svm_classifier.logRegressionLayer.params}
    svm_train_model = theano.function(inputs=[index], outputs=cost,
            updates=svm_updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})


    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.

    epoch = 0
    done_looping = False

    while (epoch < n_epochs): # and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = svm_train_model(minibatch_index)
            # iteration number
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                svm_validation_losses = [svm_validate_model(i) for i
                                         in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(svm_validation_losses)

                #svm_train_losses = [svm_train_loss(i) for i in xrange(n_train_batches)]
                #svm_train_score = numpy.mean(svm_train_losses)
                #svm_test_losses = [svm_test_model(i) for i in xrange(n_test_batches)]
                #svm_test_score = numpy.mean(svm_test_losses)
                #print('epoch %i, training error %f %%, validation error %f %%, test error %f %%' %
                #     (epoch, svm_train_score * 100., this_validation_loss * 100., svm_test_score * 100.))
                print('epoch %i, validation error %f %%' %
                      (epoch, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    svm_test_losses = [svm_test_model(i) for i
                                       in xrange(n_test_batches)]
                    svm_test_score = numpy.mean(svm_test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           svm_test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    

    # calculate current loss
    svm_train_losses = [svm_train_loss(i) for i in xrange(n_train_batches)]
    svm_train_score = numpy.mean(svm_train_losses)
    svm_validation_losses = [svm_validate_model(i) for i in xrange(n_valid_batches)]
    svm_validation_score = numpy.mean(validation_losses)
    svm_test_losses = [svm_test_model(i) for i in xrange(n_test_batches)]
    svm_test_score = numpy.mean(svm_test_losses)
    print "SVM: after training: Training: %.3f%%, Validation: %.3f%%, Test: %.3f%%" % \
        (svm_train_score * 100., svm_validation_score * 100., svm_test_score * 100.)


