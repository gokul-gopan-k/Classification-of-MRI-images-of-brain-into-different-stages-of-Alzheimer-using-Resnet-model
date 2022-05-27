import pandas as pd
from training import training_model
from config import CONFIG
from data_prepare import get_data


#get test data
x_test, y_test = get_data(mode = "test")

x_test = np.array(x_test)
x_test = np.expand_dims(x_test, axis=-1)

#prediction
resnet_model = training_model()
pred=resnet_model.predict(x_test)

labels = os.listdir(CONFIG.data_test)

#print 10 sample test results
print("10 sample test results")
test_truth = [ labels[y_test[i]] for i in range(10) ]
test_result = [ labels[np.argmax(pred[i])] for i in range(10) ]
testdata = {'test_truth': test_truth, 'test_result': test_result}
result_table = pd.DataFrame(testdata)
print(result_table)
