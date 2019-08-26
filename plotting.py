import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


sns.set(color_codes=True)


important_features = ['DG6=2', 'DL0', 'DL1=7', 'MT1A=2.0', 'DG6=1', 'DL3', 'MT1A=1.0',
       'MT6=1.0', 'FL4=2', 'DL2=nan', 'DG3=6', 'MT6=2.0', 'GN1=1.0', 'DG1',
       'FL4=1', 'MT10=2', 'MT4_1', 'GN5=1', 'MT14C_2', 'GN4=1', 'MT14C_3',
       'DG3=3', 'GN3=1', 'MT4_3', 'MT4_6', 'GN3=2', 'GN5=2', 'MT6B=nan',
       'DL1=1', 'MT2', 'MT4_2', 'MT14C_1', 'GN1=nan', 'DG3=1', 'GN4=2',
       'MT6B=1.0', 'MT10=1', 'GN2=2', 'MT4_4', 'DG6=3', 'MT6A=nan',
       'DG8a=1', 'DG5_4', 'MT7', 'IFI14_2', 'MT6A=1.0', 'GN1=2.0',
       'MT14C_4', 'IFI15_2', 'MT4_5', 'IFI14_1', 'GN2=1', 'DG4=1',
       'DL2=1.0', 'IFI15_1', 'MT5=1.0', 'IFI17_2', 'DL5=5.0', 'LN1A',
       'LN2_2', 'LN2_1', 'MT18_5', 'LN1B', 'MT18_2', 'LN2_3', 'IFI17_1',
       'LN2_4', 'GN4=4', 'MT1A=3.0', 'DL8', 'DL1=9', 'FF9', 'DL7', 'DL1=8',
       'MT18_3', 'GN5=4', 'MT17_5', 'MT15=2.0', 'DG8a=2', 'GN2=4',
       'G2P1_11=1.0', 'MT17_2', 'DL4_5', 'IFI15_3', 'DG5_6', 'IFI14_3',
       'MT14A_2', 'DL14=1', 'G2P5_11=2.0', 'MT17_4', 'DL5=6.0', 'GN3=4',
       'MT16_4', 'DG10c', 'DL15=4', 'FF16_2', 'DL4_6', 'DL1=4', 'MT17_3',
       'DL1=99', 'FL4=8', 'MT17_1', 'AA3=3', 'MT7A', 'DG12C_1', 'DG8c=0',
       'DG9a=1.0', 'MT18_4', 'GN1=4.0', 'DG10b', 'FF16_1', 'MT17_12',
       'MT17_9', 'MT17_11', 'IFI17_3', 'DG12C_2', 'AA3=1', 'MT1=1',
       'DG11c', 'DL15=1', 'DG11b', 'DL23', 'MT17_8', 'DL15=3', 'IFI15_4',
       'DL4_99', 'MT14_2=1.0', 'FB2=2', 'FB4_1', 'FB20=15.0', 'MT18_1',
       'DL21', 'DL6', 'FF13', 'FL11=99', 'DG12B_1', 'FF3=99.0', 'IFI14_4',
       'DL19', 'FB26_1', 'FL4=99', 'DL24=99', 'DG5_7', 'IFI14_7', 'DL15=2',
       'MT17_6', 'FL8_2=4', 'DL2=30.0', 'DL22', 'FL9A=11', 'FB2=3',
       'AA3=4', 'FL16=2', 'DG9c=0.0', 'DL2=2.0', 'FL8_4=4', 'FL1=4',
       'IFI15_7', 'MT17_7', 'DG12B_2', 'FL15=2', 'FL10=99', 'FB4_2',
       'IFI14_5', 'DL1=10', 'DL16', 'DL11=99', 'DG9a=2.0', 'MT8', 'DL14=2',
       'FL8_5=4', 'IFI15_5', 'FL8_1=2', 'FL14=99', 'DG8b=0', 'MT16_96',
       'DL18', 'FL6_1', 'DL1=6', 'DL12=11.0', 'FL18=99', 'FL8_5=3',
       'DL14=4', 'DL11=0', 'FB4_4', 'MT18A_2=2.0', 'DL24=2', 'FB1_1',
       'FL8_2=3', 'FL14=1', 'IFI16_2=2.0', 'DL17', 'DL20', 'FB26_6',
       'MT1A=4.0', 'GN5=3', 'FL1=2', 'AA6=6.0', 'FL8_6=4', 'FL9A=1',
       'FB23_1', 'FL8_4=3', 'DL24=3', 'FL8_1=4', 'DG5_2', 'DL26_12=1',
       'FL2=2.0', 'FB26_99', 'FL16=1', 'FL15=99', 'DG8a=3', 'FL13=1',
       'FB13=99', 'DL26_12=2', 'FL8_7=3', 'FL17=1', 'DL14=3', 'FL17=99',
       'MT16_99', 'FL13=99', 'AA6=7.0', 'FL16=99', 'FB13=0', 'FF14_1',
       'IFI18=99', 'FL18=1', 'GN3=3', 'FB1_3', 'DL26_99=1', 'FL8_6=3',
       'G2P4_11=1.0', 'IFI16_1=1.0', 'IFI18=0', 'DG9b=0.0', 'FL17=2',
       'AA3=2', 'FL11=2', 'DG4=6', 'FB4_3', 'GN4=3', 'FL8_5=2', 'FL8_2=2',
       'FL6_3', 'MT17_10', 'DG4=5', 'FL8_3=3', 'FB2=1', 'DG5_5', 'AA6=8.0',
       'MT1=2', 'FL8_7=2', 'FL8_4=2', 'FL8_6=2', 'FB24=15.0', 'GN1=3.0',
       'DG8a=4', 'DL26_99=2', 'DL1=5', 'FB26_10', 'FB26_8', 'FF2=1.0',
       'FL8_1=3', 'FL15=3', 'FB26_2', 'DG5_1', 'IFI14_6', 'FL8_3=4',
       'FL8_7=4', 'FL8_3=2', 'IFI16_1=2.0', 'FB18=5', 'DG9a=0.0',
       'IFI16_2=1.0', 'FB26_5', 'GN2=3', 'MT15=1.0', 'FB1_2', 'G2P1_9=2.0',
       'FL9B=11.0', 'DG3A=4', 'FL10=11', 'FB20=1.0', 'MT18A_4=2.0',
       'DG8b=1', 'FB19=10', 'DL25_1', 'FB26_11', 'FL6_2', 'IFI15_6',
       'MM1=1', 'DG8c=1', 'FF2=2.0', 'DL25_3', 'FF10_2', 'FF14_2',
       'FB26_4', 'G2P1_11=2.0', 'FF5=3.0', 'FL11=1', 'DG4=7', 'FL7_2=2',
       'IFI24=2.0', 'FB26_7', 'DL24=1', 'FL11=3', 'DL25_2', 'FF10_1',
       'DL14=5', 'MT16_1', 'MT17_13', 'IFI17_7', 'FF6_7=2.0', 'FL6_4',
       'FL7_1=2', 'FB26_3', 'IFI17_5', 'DG13_2', 'FL3=3.0', 'FF6_1=2.0',
       'FB19B_1=2.0', 'FB3', 'DG5_10', 'IFI17_4', 'FL14=2', 'FL1=1',
       'MT14A_7', 'FB19B_4=2.0', 'MT18_8', 'MT1=0', 'DG13_7', 'FL8_1=1',
       'GN5=99', 'AA5=5.0', 'MT1A=99.0', 'FF6_3=2.0', 'DG6=7', 'FB20=2.0',
       'FB20=14.0', 'MT16_2', 'MT16_3', 'IFI20_9', 'DG9a=3.0', 'MT1=99',
       'FF6_6=2.0', 'FF6_2=2.0', 'FF2A=1.0', 'DL24=5', 'DG3A=2',
       'FB19B_4=99.0', 'FF6_9=2.0', 'FF5=1.0', 'MT9=15.0', 'FB19B_3=99.0',
       'IFI16_1=3.0', 'FL15=1', 'FB19B_3=2.0', 'FL2=1.0', 'FB19B_2=2.0',
       'FB4_96', 'FF4', 'FB19B_2=99.0', 'FF6_10=2.0', 'FB19B_96=99.0',
       'MT18A_1=2.0', 'GN4=99', 'FB16A_8', 'IFI24=4.0', 'AA5=3.0',
       'FF6_4=2.0', 'FB19B_1=99.0', 'MT18_6', 'FL9B=6.0', 'DG3=5',
       'FL9A=3', 'DG3=99', 'IFI24=10.0', 'FB18=1', 'MT18_96', 'DG13_5',
       'G2P1_12=2.0', 'DL24=9', 'FL8_4=5', 'FL2=3.0', 'DL4_2', 'FF6_8=2.0',
       'FL7_1=99', 'MT6=3.0', 'FF19_4', 'DG6=99', 'FL7_4=2', 'G2P1_13=2.0',
       'MT14A_11', 'FL9C=11.0', 'FB19B_5=99.0', 'FL11=4', 'FF6_2=99.0',
       'FL10=1', 'DG8a=99', 'FB17_8', 'FL9B=2.0', 'IFI22_1', 'IFI16_2=3.0',
       'G2P1_1=2.0', 'G2P1_7=2.0', 'DG9b=1.0', 'FF6_5=2.0', 'DL26_5=2',
       'DG8a=5', 'FL8_3=5', 'FL3=8.0', 'FB19B_96=2.0', 'DL26_5=1', 'DG4=4',
       'IFI22_7', 'FL7_3=2', 'G2P1_3=2.0', 'FF19_1', 'FL3=2.0', 'DL25_4',
       'FL8_7=5', 'FL10=2', 'FL1=3', 'G2P1_8=2.0', 'MM1=2', 'FB26_9',
       'DL1=2', 'FL7_5=2', 'FB16A_1', 'G2P1_4=2.0', 'DG13_96', 'FF3=2.0',
       'FB19B_5=2.0', 'FB26_96', 'FL7_4=99', 'FL8_7=1', 'DG13_1',
       'IFI20_4', 'DL5=2.0', 'G2P1_99=2.0', 'FL7_6=99', 'FL12=1',
       'FL7_3=99', 'FF2A=2.0', 'G2P1_14=2.0', 'G2P1_6=2.0', 'FF6_7=99.0',
       'FL9C=6.0', 'FF6_9=99.0', 'DL14=6', 'DG4=99', 'DL24=4', 'IFI20_5',
       'G2P1_10=2.0', 'DG8b=2', 'FL9A=7', 'DG13_3', 'FL8_6=5', 'FL8_2=5',
       'FF6_10=99.0', 'IFI24=6.0', 'DL27=5.0', 'GN3=99', 'DL4_17',
       'FB24=1.0', 'GN2=99', 'FB17_1', 'DL25_5', 'FL7_2=99', 'FB19=1',
       'FL9B=7.0', 'FL8_3=1', 'FL8_5=1', 'DG4=3', 'FF6_8=99.0', 'DG14',
       'DL25_7', 'G2P1_5=2.0', 'FF6_2=1.0', 'FB20=6.0', 'DL1=3', 'FL12=99',
       'IFI16_2=7.0', 'FL8_5=5', 'FF6_6=99.0', 'MT6=4.0', 'FF6_8=1.0',
       'FL9B=1.0', 'FL7_6=2', 'IFI16_2=99.0', 'FL9B=3.0', 'FB22_8',
       'DL25_6', 'G2P1_96=2.0', 'MT18A_1=1.0', 'FF6_3=99.0', 'FL3=4.0',
       'G2P1_2=2.0', 'IFI24=1.0', 'DG9c=1.0', 'FF2A=12.0']

to_delete = ['MT6C', 'DG1']

name = 'train_combined_1_small'
file = name + '.csv'
df = pd.read_csv(file, low_memory=False)
print("df train shape ", df.shape)

for key in ['DL2=nan']:
	sns.countplot(y=key, hue="is_female", data=df, palette="pastel")
	plt.show()
