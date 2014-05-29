from jobman.tools import DD
import numpy
import json
import theano
import scipy.io.wavfile
from pylearn2.config import yaml_parse

def reconstruct_phase( model ):
    valid_dataset = model.monitor._datasets[0]
    if isinstance( valid_dataset, basestring ):
        valid_dataset = yaml_parse.load( valid_dataset )
    X = model.get_input_space().make_batch_theano()
    y = model.fprop(X)
    f = theano.function([X], y)
    amps = valid_dataset.X[0:100,:]
    phases = f( amps )
    l = phases.shape[1]/2
    norms = numpy.sqrt( phases[:,:l]**2 + phases[:,l:]**2 )
    phases = phases / numpy.hstack( (norms,norms) )
    
    return valid_dataset.to_audio( amps, phases ).ravel()

def results_extractor(train_object):
    channels = train_object.model.monitor.channels
    train_obj_history = map( lambda x: float(x), list( channels['train_objective'].val_record ) )
    valid_obj_history = map( lambda x: float(x), list( channels['valid_objective'].val_record ) )

    train_history =  ('graph', 'Train objective', 'epochs', { 'train': train_obj_history,
                                                              'valid': valid_obj_history } ) 
    latest_train_obj = int(1000*train_history[3]['train'][-1])/1000.0
    latest_valid_obj = int(1000*train_history[3]['valid'][-1])/1000.0
    best_valid_obj = int(1000*numpy.min( train_history[3]['valid'] ))/1000.0
    best_valid_epoch = numpy.argmin( train_history[3]['valid'] )
    #examples_seen = 0#train_obj.channels['examples_seen']
    #epochs_seen = 0#channels['epochs_seen']
    total_seconds_last_epoch = int(10*channels['total_seconds_last_epoch'].val_record[-1])/10.0
    
    #reconstructed_phase = ('sound', reconstruct_phase( train_object ) )
    #f = open('reconstructed_phase.wav')
    scipy.io.wavfile.write('reconstructed_phase.wav', 16000, numpy.array( reconstruct_phase( train_object.model ) ) )
    

    return DD(best_valid_bpc=best_valid_obj,
              best_valid_epoch=best_valid_epoch,
              latest_train_obj=latest_train_obj,
              latest_valid_obj=latest_valid_obj, 
              train_history=json.dumps(train_history),
              #examples_seen=examples_seen,
              #epochs_seen=epochs_seen,
              total_seconds_last_epoch=total_seconds_last_epoch )
              #reconstructed_phase = json.dumps(reconstructed_phase) )
