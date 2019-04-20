"""
Computes the F1 score on tagged data

@author: N. Reimers, Ines
"""


# Method to compute the accuracy. Call predict_labels to get the labels for the dataset
def compute_f1(predictions, dataset_y, idx2Label, epoch, dev_or_test):
    #
    # y_prob = model.predict(x) => predictions
    y_classes = predictions.argmax(axis=-1)
    # predicted_label = sorted(labels)[y_classes]

    # for e in predictions:
    # print e
    label_y = [idx2Label[element] for element in dataset_y]
    pred_labels = [idx2Label[element] for element in y_classes]

    prec, rec = compute_precision_recall(pred_labels, label_y, epoch, idx2Label)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    log(prec, rec, f1, dev_or_test)

    return prec, rec, f1


def compute_precision_recall(guessed, correct, epoch, idx2Label):
    tp_counts = {}
    guessed_label_counts = {}
    correct_label_counts = {}
    outfile = "out." + str(epoch)
    file = open(outfile, 'w')

    for label in idx2Label.values():
        tp_counts[label] = 0
        guessed_label_counts[label] = 0
        correct_label_counts[label] = 0

    for i in range(len(guessed)):
        if guessed[i] == correct[i]:
            tp_counts[correct[i]] += 1
        file.write(str(guessed[i]) + "\t" + str(correct[i]) + "\n")  # write correct and guessed labels to file

        guessed_label_counts[guessed[i]] += 1
        correct_label_counts[correct[i]] += 1

    sum_p = 0.0
    sum_r = 0.0
    sum_n = 0.0
    for label in tp_counts.keys():
        if correct_label_counts[label] == 0:
            continue
        elif guessed_label_counts[label] == 0:
            p_i = 0.0
            r_i = float(tp_counts[label]) / correct_label_counts[label]
            sum_p += p_i * correct_label_counts[label]
            sum_r += r_i * correct_label_counts[label]
            sum_n += correct_label_counts[label]
        else:
            p_i = float(tp_counts[label]) / guessed_label_counts[label]
            r_i = float(tp_counts[label]) / correct_label_counts[label]
            sum_p += p_i * correct_label_counts[label]
            sum_r += r_i * correct_label_counts[label]
            sum_n += correct_label_counts[label]

    precision = sum_p / sum_n
    recall = sum_r / sum_n

    file.write("precision: " + str(precision))
    file.close()
    return precision, recall


def log(prec, rec, f1, dev_or_test):
    with open("LOG_precision" + dev_or_test, "a") as pl:
        pl.write(str(prec)+"\n")
        pl.close()
    with open("LOG_recall" + dev_or_test, "a") as rl:
        rl.write(str(rec)+"\n")
        rl.close()
    with open("LOG_f1" + dev_or_test, "a") as fl:
        fl.write(str(f1)+"\n")
        fl.close()

