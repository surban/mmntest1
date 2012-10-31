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

    # make trainset smaller
    tsize = 2
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
    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
    validation_score = numpy.mean(validation_losses)
    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_score = numpy.mean(test_losses)
    print "Validation: %.3f%%, Test: %.3f%%" % (validation_score * 100.,
                                                test_score * 100.)

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
    errs = 0
    werrs = 0
    for n in range(hl.n_out):
        print "Hidden neuron: %d\r" % n,
        probs.append(svmutil.svm_problem(svm_targets[n], svm_inputs))
        params.append(svmutil.svm_parameter("-q -s 0 -t 0 -c 100"))
        svms.append(svmutil.svm_train(probs[n], params[n]))
        svmutil.svm_save_model("hidden%04d.svm" % n, svms[n])

        # test model
        predictions, acc, _ = svmutil.svm_predict(svm_targets[n], svm_inputs, 
                                                  svms[n])
        #print "Accuracy: %f" % acc[0]
        for prediction, label in zip(predictions, svm_targets[n]):
            if prediction != label:
                errs += 1

        # get weights from SVM
        w, b = get_svm_weights(svms[n], hl.n_in)
        ws.append(w)
        bs.append(b)

        # check calculated weights
        #for i in range(len(svm_inputs)):
        #    #print "Predicting with libsvm for input %d" % i
        #    # predict using direct libsvm
        #    xi, idx = svm.gen_svm_nodearray(svm_inputs[i], isKernel=False)
        #    dec_values = (svm.c_double * 2)()
        #    label = svm.libsvm.svm_predict_values(svms[n], xi, dec_values)

        #    # predict using weights
        #    #print len(svm_inputs[i])
        #    predv = numpy.dot(w, svm_inputs[i]) + b
        #    pred = numpy.sign(predv)

        #    if pred != label:
        #        print "Error prediction, no match with libsvm"
        #        print "target: %f, prediction: %f, direct prediction: %f" % \
        #            (svm_targets[n][i], pred, label)
        #        print "predv: %f" % predv

        #        # calculate scalar products
        #        sprods = []
        #        ssum = 0
        #        for j in range(len(fullsvs)):
        #            sprods.append(numpy.dot(svm_inputs[i], fullsvs[j]))
        #            print "fullsvs[j]:", fullsvs[j]
        #            print "sprods[%d] = input[%d] * sv[%d] = %f" % (j, i, j, sprods[j])
        #            print "coeff[%d] = %f" % (j, svcoef[j][0])
        #            print "coeff[%d] * sprods[%d] = %f" % (j, j, sprods[j] * svcoef[j][0])
        #            ssum += sprods[j] * svcoef[j][0]
        #        print "sum(sprods * coeff) = %f" % ssum

        #        raise Exception()

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
        print "Neuron %d: Positive: %d, Negative: %d" % (n, pos, neg)
        if werrs > 1:
            raise Exception()

    print "Done, SVM model errors: %d, my predictor errors: %d" % (errs, werrs)    

    # construct weight and bias matrix
    svm_hidden_W = numpy.zeros((hl.n_in, hl.n_out))
    for n in range(hl.n_out):
        svm_hidden_W[:,n] = ws[n]
    svm_hidden_b = numpy.asarray(bs)

    # construct model using SVMs
    svm_classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=500, n_out=10)
    svm_classifier.logRegressionLayer.W.set_value(params['logreg_W'])
    svm_classifier.logRegressionLayer.b.set_value(params['logreg_b'])
    svm_classifier.hiddenLayer.W.set_value(svm_hidden_W)
    svm_classifier.hiddenLayer.b.set_value(svm_hidden_b)

    # test model functions
    svm_training_loss = theano.function(inputs=[index],
            outputs=svm_classifier.errors(y),
            givens={
                x: training_set_x[index * batch_size:(index + 1) * batch_size],
                y: training_set_y[index * batch_size:(index + 1) * batch_size]})

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
    svm_training_losses = [svm_training_loss(i) for i in xrange(n_valid_batches)]
    svm_training_score = numpy.mean(svm_training_losses)
    svm_validation_losses = [svm_validate_model(i) for i in xrange(n_valid_batches)]
    svm_validation_score = numpy.mean(validation_losses)
    svm_test_losses = [svm_test_model(i) for i in xrange(n_test_batches)]
    svm_test_score = numpy.mean(svm_test_losses)
    print "SVM: Training: %.3f%%, Validation: %.3f%%, Test: %.3f%%" % \
        (svm_training_score * 100., svm_validation_score * 100., svm_test_score * 100.)

    


