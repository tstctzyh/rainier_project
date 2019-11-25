import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def main():
	with open('rainier_model.pickle','rb') as f:
            tl2 = pickle.load(f)

    result=tl2.predict(["13.643750","26.321667","19.715000","27.839583","68.004167","88.496250","2"])
    print(result)

main()
