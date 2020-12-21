from json import loads, dumps
import torch

def lineprint(msg='', ch = '-'):
    for _ in range(80):
        print(ch, end='')
    print('\n\t' + msg)

def myprint(st):
    print("\t"+st)

def loadTrainingLog(n_hidden, mode):
    with open(f"training_logs/epch40_h{n_hidden}_mode{mode}_traintest.txt", 'r') as f:
        st = f.read()
        return loads(st)

def save_network(basename, network):
    torch.save(network, basename+".savednn")

def load_network(basename):
    return torch.load(basename+".savednn")

def load_training_results(basename):
    autoencoder = load_network(basename)
    with open(basename+"_training.results", 'r') as f:
        st = f.read()
        ret_dictionary = loads(st)
    ret_dictionary["autoencoder"] = autoencoder
    return ret_dictionary

def save_training_results(basename, results_dictionary):
    save_network(basename, results_dictionary["autoencoder"])
    with open(basename+"_training.results",'w') as f:
        d_copy = results_dictionary
        del d_copy["autoencoder"]
        f.write(dumps(d_copy))
