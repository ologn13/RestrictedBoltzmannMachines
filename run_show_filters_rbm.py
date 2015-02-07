import mlProblems_generic as mlpb
from rbm import RBM

print "Loading dataset..."
import ocr_letters as mldataset
load_to_memory = True

dataset_dir = '/home/vikrant/Desktop/OCR_Dataset_RBM'
#mldataset.obtain(dataset_dir)
all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory)

train_data, train_metadata = all_data['train']
valid_data, valid_metadata = all_data['valid']
test_data, test_metadata = all_data['test']

trainset = mlpb.ClassificationProblem(train_data,train_metadata)
validset = trainset.apply_on(valid_data,valid_metadata)
testset = trainset.apply_on(test_data,test_metadata)
#trainset,validset,testset = dataset_store.get_classification_problem('ocr_letters')

print "Train RBM for 10 iterations... (this might take a few minutes)"
rbm = RBM(n_epochs = 10,
          hidden_size = 200,
          lr = 0.01,
          CDk = 1,
          seed=1234
          )

rbm.train(mlpb.SubsetFieldsProblem(trainset))
rbm.show_filters()

