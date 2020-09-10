import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from numpy import nan as NA
import matplotlib.pyplot as plt

df1 = pd.read_csv('f4-5_sharp2.csv')
df1.plot()
plt.ylim([-500,500])
plt.show()