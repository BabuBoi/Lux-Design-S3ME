import pickle

def save_data(training_data, filename="training_data.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(training_data, f)

def load_data(filename="training_data.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)