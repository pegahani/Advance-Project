import matplotlib.pyplot as plt

# y_v = [ 137,  357,  701,  1154,  1800, 2758, 4097, 5810, 8010, 11486]
# #y_v = [i/10000.0 for i in y_v]
# y_opt = [ 0.0693014383316, 0.0641954183578, 0.0662003755569, 0.0715884208679,0.0718366384506,  0.116097545624,
#          0.0933467626572,  0.108767294884,0.101793313026,0.118519616127  ]
#
# import numpy as NP
# from matplotlib import pyplot as PLT
#
# # just create some random data
# fnx = lambda : NP.random.randint(3, 10, 10)
# y = NP.row_stack((y_v, y_opt))
#
# print "y", y
#
# # this call to 'cumsum' (cumulative sum), passing in your y data,
# # is necessary to avoid having to manually order the datasets
# x = NP.arange(50, 501, 50)
# y_stack = NP.cumsum(y, axis=0)   # a 3x10 array
#
# fig = PLT.figure()
# ax1 = fig.add_subplot(111)
#
# ax1.fill_between(x, 0, y_opt, facecolor="red", alpha=.7)
# ax1.fill_between(x, y_opt, y_v, facecolor="#1DACD6", alpha=.7)
#
# PLT.show()
#
# #---------------------------------------------------------------------------
#
# dg = pandas.DataFrame(columns=['0.1','0.01','0.001'], index=range(50,501,50))
# print dg
#
# # dg['0.001'] = [154.203951621,  376.211200428, 754.875621605,1203.43353803, 1915.28135231, 2903.19739506, 4292.14182487,
# #                6042.75854566, 8688.68465037, 12182.3804761]
# #
# # dg['0.01'] = [ 137.915304756,  357.180779433,  701.924669957,  1154.56340885,  1800.09437952, 2758.27340014, 4097.0526026,
# #                 5810.52119086,  8010.90948367, 11486.5473624]
# #
# # dg['0.1'] = [ 118.646293616,336.642904639,  652.146585989,  1111.45338237,1658.21103683, 2677.13862669, 3902.84204473,
# #              5861.78221564, 8028.554318, 11376.5021693 ]
#
#
# dg['0.001'] = [154,  376, 754, 1203, 1915, 2903, 4292, 6042, 8688, 12182]
#
# dg['0.01'] = [ 137,  357,  701,  1154,  1800, 2758, 4097, 5810, 8010, 11486]
#
# dg['0.1'] = [ 118, 336,  652,  1111, 1658, 2677, 3902, 5861, 8028, 11376 ]
#
#
# dg.plot(kind='bar')
#
#
# plt.xlabel('number of states')
# plt.ylabel('time(second)')
# plt.title('V-bar propagation time vs. number of states')
# plt.legend(loc = 2)
#
#
# plt.show()

#------------------------------------------------------------


fig, ax = plt.subplots()
ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,9.5,10], minor=False)
ax.yaxis.grid(True, which='major')
ax.yaxis.grid(True, which='minor')



states_d_2 = range(50, 501, 50)
error_d_2_001 = [ 2.1288746304, 0.699205470197, 2.20697229935, 2.42178296607, 1.63887410235, 1.30640086343, 2.20372187839,
                  0.742265778194,1.8535160372, 1.73156786233 ]
error_d_2_01 = [1.84063306886, 1.73309366217, 1.29385750906, 1.01195876516,0.813906593909,  1.29448481692, 2.65010715022,
               2.17740223129,0.987535643835, 0.547733995498  ]
error_d_2_1 = [1.82175558049,0.72791676579, 3.65462218669, 0.598349928332, 2.10794703912,1.55611397138, 1.64532852238,
               1.67739026812, 2.13640289256, 1.56934897353 ]


plt.plot(states_d_2, error_d_2_001, marker='s', label='epsilon=0.001')
plt.plot(states_d_2, error_d_2_01, marker='o', linestyle='--', color='red',label='epsilon=0.01')
plt.plot(states_d_2, error_d_2_1, marker='*', linestyle='-.', color='green',label='epsilon=0.1')

plt.ylim(0,10)
plt.xlim(0,600)

plt.xlabel('number of states')
plt.ylabel('error')
plt.title('error vs. number of states')
plt.legend(loc = 1)
plt.show()