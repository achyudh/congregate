import pandas as pd

from lib.preprocess import extract_features


def classic4():
    data = pd.read_table("data/classic4/classic4-stemmed.txt")
    data.columns = ["labels", "text"]
    classes = {'cran': 0, 'med': 1, 'cisi': 2, 'cacm': 3}

    features, labels = extract_features(classes, data)
    return features, labels


def reuters8():
    data = pd.read_table("data/r8/r8-stemmed.txt")
    data.columns = ["labels", "text"]
    classes = {
        'acq': 0,
        'crude': 1,
        'earn': 2,
        'grain': 3,
        'interest': 4,
        'money-fx': 5,
        'ship': 6,
        'trade': 7
    }

    features, labels = extract_features(classes, data)
    return features, labels


def webkb():
    data = pd.read_table("data/webkb/webkb-stemmed.txt")
    data.columns = ["labels", "text"]
    classes = {'project': 0, 'faculty': 1, 'course': 2, 'student': 3}

    features, labels = extract_features(classes, data)
    return features, labels


def ng20():
    data = pd.read_table("data/20ng/20ng-stemmed.txt")
    data.columns = ["labels", "text"]
    classes = {
        'alt.atheism': 0,
        'comp.graphics': 1,
        'comp.os.ms-windows.misc': 2,
        'comp.sys.ibm.pc.hardware': 3,
        'comp.sys.mac.hardware': 4,
        'comp.windows.x': 5,
        'misc.forsale': 6,
        'rec.autos': 7,
        'rec.motorcycles': 8,
        'rec.sport.baseball': 9,
        'rec.sport.hockey': 10,
        'sci.crypt': 11,
        'sci.electronics': 12,
        'sci.med': 13,
        'sci.space': 14,
        'soc.religion.christian': 15,
        'talk.politics.guns': 16,
        'talk.politics.mideast': 17,
        'talk.politics.misc': 18,
        'talk.religion.misc': 19
    }

    features, labels = extract_features(classes, data)
    return features, labels
