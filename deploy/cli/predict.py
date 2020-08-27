def predict():
    raise NotImplementedError

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Consume trainend model.")
    args = vars(ap.parse_args())

    predict()

    