

from sklearn.model_selection import train_test_split
import numpy as np

def max_depth_search(model, X, y, cycles=20, depth_list=None, verbose=False):
    
    if depth_list is None:
        depth_list = [1, 3, 6, 10, None]

    highest_acc = 0
    opt_depth = 1

    for depth in depth_list:
        
        accuracy_list = []
        for _ in range(cycles):

            X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8)
            tree_classifier = model(max_depth= depth)
            tree_classifier.fit(X_train, y_train)
            y_hat = tree_classifier.predict(X_validation).round()

            accuracy_list.append(np.mean(y_hat == y_validation))

        accuracy = np.mean(accuracy_list)
        
        if accuracy > highest_acc:
            highest_acc = accuracy
            opt_depth = depth
        
        if verbose:
            print('max depth=', depth, ':')
            print('average accuracy:', accuracy)
            print('accuracy spread:', np.max(accuracy_list) - np.min(accuracy_list))

    if verbose:
        print('')
        print('Best cycle:') 
        print('Max depth=', opt_depth)
        print('Accuracy=', highest_acc)

    return opt_depth, highest_acc




def learning_rate_search(model, X, y, cycles=20, lr_list=None, verbose=False):
    
    if lr_list is None:
        lr_list = [1, 0.3, 0.1, 0.03, 0.01]

    highest_acc = 0
    opt_lr = 1

    for lr in lr_list:
        
        accuracy_list = []
        for _ in range(cycles):

            X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8)
            tree_classifier = model(learning_rate= lr)
            tree_classifier.fit(X_train, y_train)
            y_hat = tree_classifier.predict(X_validation).round()

            accuracy_list.append(np.mean(y_hat == y_validation))

        accuracy = np.mean(accuracy_list)
        
        if accuracy > highest_acc:
            highest_acc = accuracy
            opt_lr = lr
        
        if verbose:
            print('Learning Rate=', lr, ':')
            print('average accuracy:', accuracy)
            print('accuracy spread:', np.max(accuracy_list) - np.min(accuracy_list))

    if verbose:
        print('')
        print('Best cycle:') 
        print('Learning Rate=', opt_lr)
        print('Accuracy=', highest_acc)

    return opt_lr, highest_acc

def max_features_search(model, X, y, cycles=20, max_features_list=None, max_depth=None, verbose=False):
    
    if max_features_list is None:
        max_features_list = [i + 1 for i in range(X.shape[1] -1)]
        max_features_list.append('sqrt')
        max_features_list.append('log2')
        max_features_list.append('auto')

    highest_acc = 0
    opt_features = 1

    for features in max_features_list:
        
        accuracy_list = []
        for _ in range(20):
            X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8)
            rf_c = model(max_features= features, max_depth=max_depth)
            rf_c.fit(X_train, y_train)
            y_hat = rf_c.predict(X_validation).round()

            acc = 1 - np.mean(y_hat != y_validation)
            accuracy_list.append(acc)
        accuracy = np.mean(accuracy_list)

        if accuracy > highest_acc:
            highest_acc = accuracy
            opt_features = features

        if verbose:
            print('Max Features=', features, ':')
            print('average accuracy:', accuracy)
            print('accuracy spread:', np.max(accuracy_list) - np.min(accuracy_list))

    if verbose:
        print('')
        print('Best cycle:') 
        print('Max Features=', opt_features)
        print('Accuracy=', highest_acc)

    return opt_features, highest_acc