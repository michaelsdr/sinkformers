import argparse
import os
import trainer_cats_and_dogs

parser = argparse.ArgumentParser()
parser.add_argument("--n_it", type=int, default='3')
parser.add_argument("--seed", type=int, default='0')
args = parser.parse_args()


n_it = args.n_it
seed = args.seed

save_dir = 'results'
save_model_dir = 'results_model'

try:
    os.mkdir(save_dir)

except:
    pass

try:
    os.mkdir(save_model_dir)

except:
    pass


save_adr = save_dir + '/%d_it_%d.npy' % (n_it, seed)
save_model = save_model_dir + '/%d_it_%d.pth' % (n_it, seed)
res = trainer_cats_and_dogs.main(n_it, save_adr, save_model, seed=seed)

