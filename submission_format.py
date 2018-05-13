import numpy as np

def submission_format(ids, labels):
    # transform final predictions into correct submission format

    ids = np.array(ids)
    labels = np.array(labels)

    [r, c] = labels.shape
    with open('submission.csv', 'wb') as f:
        f.write("image_id, label_id")
        f.write('\n')
        for i in range(r):
            lab = []
            for l in range(c):
                if labels[i, l] == 1:
                    lab.append(l + 1)
            l = ' '.join(str(v) for v in lab)
            data = '{},{}'.format(ids[i], l)
            f.write(data)
            f.write('\n')
