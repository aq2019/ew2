from dataset_for_comp_status_model import StatusDataset
from status_model import SimpleTreeModel

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix, f1_score
import argparse

parser = argparse.ArgumentParser(description='Train completion status models')

parser.add_argument('-s', '--statustype', help='which status to classify: init or comp', required=True)

args = parser.parse_args()

which = args.statustype

bucket = '[redacted_bucket_name]'
init_model_key = 'saved_model/init_simple_tree.pkl'
comp_model_key = 'saved_model/comp_simple_tree.pkl'


tree_params = {'max_depth':[d for d in range(4,15,2)],
              'min_samples_split':[s for s in range(2, 7)]}
valid_ratio = 0.2

if __name__ == '__main__':
    data = StatusDataset()
    if which == 'init':
        val_size = int(data.init_train_x.shape[0]*(1-valid_ratio))
        train_x, val_x = data.init_train_x.iloc[:val_size, :], data.init_train_x.iloc[val_size:, :]
        train_y, val_y = data.init_train_y[:val_size], data.init_train_y[val_size:]
        init_best_f1 = 0
        init_best_f1_conf = None
        for g in ParameterGrid(tree_params):
            simple_tree = SimpleTreeModel()
            simple_tree.fit_model(train_x, train_y, **g)
            val_ypred = simple_tree.pred(val_x)
            val_cm = confusion_matrix(val_y, val_ypred)
            print(val_cm)
            f1 = f1_score(val_y, val_ypred, average='weighted')
            print('f1: {0}'.format(f1))
            if f1 >= init_best_f1:
                init_best_f1 = f1
                init_best_f1_conf = g
            print('********'*5) 
        print('best init f1: {0}'.format(init_best_f1))
        print('best init f1 conf: {0}'.format(init_best_f1_conf))
        init_simple_tree_final = SimpleTreeModel()
        init_simple_tree_final.fit_model(data.init_train_x, data.init_train_y, **init_best_f1_conf)
        init_simple_tree_final.save_model(bucket, init_model_key)
        print('initiated only status classification simple tree model saved to s3')

    if which == 'comp':
        val_size = int(data.comp_train_x.shape[0]*(1-valid_ratio))
        train_x, val_x = data.comp_train_x.iloc[:val_size, :], data.comp_train_x.iloc[val_size:, :]
        train_y, val_y = data.comp_train_y[:val_size], data.comp_train_y[val_size:]
        comp_best_f1 = 0
        comp_best_f1_conf = None
        count = 1
        for g in ParameterGrid(tree_params):
            simple_tree = SimpleTreeModel()
            simple_tree.fit_model(train_x, train_y, **g)
            val_ypred = simple_tree.pred(val_x)
            val_cm = confusion_matrix(val_y, val_ypred)
            print(val_cm)
            f1 = f1_score(val_y, val_ypred, average='weighted')
            print('f1: {0}'.format(f1))
            if f1 >= comp_best_f1:
                comp_best_f1 = f1
                comp_best_f1_conf = g
            print('********'*5) 
        print('best comp f1: {0}'.format(comp_best_f1))
        print('best comp f1 conf: {0}'.format(comp_best_f1_conf))

        comp_simple_tree_final = SimpleTreeModel()
        comp_simple_tree_final.fit_model(data.comp_train_x, data.comp_train_y, **comp_best_f1_conf)
        comp_simple_tree_final.save_model(bucket, comp_model_key)
        print('complete status classification simple tree model saved to s3')


