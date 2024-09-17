# Add more imports as/if needed...
from data import class_type, data_train, features

# Allowed imports: Modules included in Python (see https://docs.python.org/3.10/py-modindex.html),
# and external libraries listed in requirements.txt

# Modify the code below to return "sensible" classes based on input features.
# The final classifier must return integers in the range 1-7 (corresponding to 
# 7 animal classes). Don't expect to get to 100% correct classification. 
# It's most important that your code makes sense, has at least some correct
# classifications, and that you document your thinking/method in the report. 
def classify_animal(hair, feathers, eggs, milk, airborne, aquatic, predator, toothed,
                    backbone, breathes, venomous, fins, legs, tail, domestic, catsize):
    """ Classifies animal based on 16 features 

    Arguments:
    ----------
    hair, feathers, ... : int
        Features descibing an animal

    Returns:
    ---------
    class_int: int
        Integer in range 1-7 corresponding to 7 classes of animal
    """
    variables = [hair, feathers, eggs, milk, airborne, aquatic, predator, toothed,
                 backbone, breathes, venomous, fins, legs, tail, domestic, catsize]

    knn = [] #Nearest neighbour array 
    number_of_neighbours = 3 # How many nearest neighbours to check
    for animal in data_train:
        knn_distance = 0
        # for feature in animal[1].length() - 1:
        # for feature in (data_train[animal][:-1]):
        for feature in range(0, len(data_train[animal][:-1])):
            #print(feature)
            #knn_distance += (variables[feature] - feature)**2
            knn_distance += (variables[feature] - data_train[animal][feature])**2
        # print(knn_distance)
        from math import sqrt
        knn_distance = sqrt(abs(knn_distance))
        # print('Total distance: %s' % knn_distance)
        if len(knn) < number_of_neighbours:
            knn.append([knn_distance, animal])
            # knn.append([knn_distance, animal[-1]])
            knn = sorted(knn, key=lambda x: x[0])
            # import operator
            # knn = sorted(knn, key=operator.itemgetter(1))
        else:
            # print(knn)
            for stored_neighbour in knn:
                # print(stored_neighbour[0])
                # print(knn_distance)
                if stored_neighbour[0] < knn_distance:
                    knn = knn[:-1]
                    # print(data_train[animal][-1])
                    # print('LAST ANIMAL LIST: {} from {}'.format(animal, data_train[animal][-1]))
                    knn.append([knn_distance, animal])
                    # knn = sorted(knn, key=lambda x: x[0])
                    import operator
                    knn = sorted(knn, key=operator.itemgetter(1))
    knn_calc = {}
    # print(knn)
    for stored_neighbour in knn:
        # print(stored_neighbour)
        #print(stored_neighbour[0])
        #knn_calc[stored_neighbour[1]] += 1
        # print(stored_neighbour)
        if data_train[stored_neighbour[1]][-1] not in knn_calc:
            knn_calc[data_train[stored_neighbour[1]][-1]] = 1
        else:
            knn_calc[data_train[stored_neighbour[1]][-1]] += 1
    ret_val = 0
    # print(knn_calc)
    for x in knn_calc:
        # print(x)
        # print('Sjekk forskjell: {} og {}'.format(x, knn_calc[x]))
        if ret_val == 0:
            ret_val = x
        elif knn_calc[x] > knn_calc[ret_val]:
            ret_val = x
    # print('Return: ', ret_val)
    return ret_val



# Helper function for showing data (not part of assignment)
def print_data():
    """ Print animal name and feature names and values """
    for animal_name, animal_features in data_train.items():
        feature_string = ', '.join([f'{feature_name} = {feature_value}'
                                    for feature_name, feature_value 
                                    in zip(features, animal_features)])
        print(f'{animal_name}: ' + feature_string)


def run_classifier():
    """ Run classifier for every animal (row) in training dataset """
    n_correct_classifications = 0
    for data_row in data_train.values():
        class_int = data_row[-1]
        animal_features = data_row[:-1]
        if classify_animal(*animal_features) == class_int:
            n_correct_classifications += 1
    print('Number of correct classifications: ' +
          f'{n_correct_classifications} / {len(data_train)}')


if __name__ == "__main__":
    # print_data()
    run_classifier()
