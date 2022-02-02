from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

df_pivot = pd.read_csv('train_pivot.csv')
#f_pivot = df_pivot('ImageId','ClassId','EncodedPixels')
#df_pivot_bool = df_pivot.astype(bool)



df_defect_count = df_pivot_bool.any(axis=1).replace(True,'defect').replace(False,'no defect').value_counts()

#sns.barplot(x=df_defect_count.index,y=df_defect_count.values)

#plt.show()