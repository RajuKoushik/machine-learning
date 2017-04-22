#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    error = []

    for j in range(len(predictions)):
        error[j] = predictions[j] - net_worths[j]

    sorted_error = error.sort()

    sorted_predictions = [x for (y, x) in sorted(zip(error, predictions))]

    sorted_net_woths = [x for (y, x) in sorted(zip(error, net_worths))]

    ### your code goes here
    for i in range(0.9 * (len(predictions))):
        tup1 = (predictions[i], ages[i], predictions[i] - net_worths[i])
        cleaned_data.append(tup1)

    return cleaned_data

