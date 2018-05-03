from sklearn.preprocessing import MultiLabelBinarizer

def label_to_binary_matrix(labels):

    # initialize multilabel bainrizer
    mlb = MultiLabelBinarizer()
    bmatrix = mlb.fit_transform(labels)
    return bmatrix
