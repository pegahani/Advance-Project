import numpy as np
from matplotlib import pyplot as plt

#na = 5, d =2. get average on 10 iterations.
# in designed algorithm, we extend nodes 10 times.
states_d_2 = range(10,341,10)
queries_d_2 = [1.06,1.16666666667,1.2, 1.21,1.27551020408, 1.26, 1.36734693878,1.38, 1.36, 1.4, 1.36,  1.35555555556,
                1.32, 1.25, 1.31, 1.33, 1.43, 1.35, 1.33333333333, 1.21, 1.31578947368, 1.43, 1.39,1.39583333333,
                 1.47368421053, 1.48, 1.47, 1.55952380952, 1.45, 1.41, 1.4880952381, 1.36]

time_gather_v_bar_d_2 = [0.524569950104, 1.08781855051,1.68963897705, 2.34293433428, 3.03351721137, 4.21703208208,
                         4.94387784825,6.18291929033,  6.98000757694,7.71034476757,  9.06098522902, 11.0969394994,
                        12.4001685119,  13.1810174329, 15.2035304594, 17.2537272515,  19.5270588756, 21.0831551147,
                          23.9113519812,
                          24.8599862671,26.9700331307,  28.4650188637,  30.7086849237,  34.0831492829, 38.116236794,
                          40.8187146039, 42.7500039314, 45.9859203053, 50.8296080899,54.3897665379,   56.4330731535,
                         59.2485120225,  62.8507998239, 69.5634157848 ]

time_find_V_optimal_d_2 = [ 0.639066853523, 0.500039747783,  0.596548006535,  0.936582067013, 0.816736447568, 1.06833363295,
                             0.876011099504, 1.02868367944, 0.811032075882, 0.909110078812,0.981942131519,  0.985169565678,
                            0.870308711529,0.8058191114,  1.03357455969,0.820591421231,   0.809653656483,0.873273704052,
                              1.2771760726,0.838119797707,  1.12642886049,  0.942085039616,0.840074217947,  1.19554037571,
                             1.01434709311,1.11672084282, 1.52205367841, 1.87194385052, 1.97111473322, 2.06725628319,
                            2.43690082312,1.85145823717, 1.40359193087,  1.8773242569]

error_d_2 = [ 1.82509372326, 2.50738629208, 1.47569090962,2.01240716389,2.72793750728, 2.0546622822, 2.57316030915,
              1.64160625237, 1.88545485903,1.69158939208, 1.06972941885, 1.63892753166, 1.1541501215, 2.19426367022,
               2.72396611579,1.51664271763,  2.1265191281,  2.31905934811, 1.16044156754,  2.03300854583, 3.66750030135,
              2.62800096611,  2.1433289305, 2.48553386684,2.08634766597, 3.45836682211,1.66422176211, 2.68777439589,
              2.13971978479,   0.357111644167, 2.1102163137, 1.60203151257,  2.58531147493,  2.43545139376]

print len(time_gather_v_bar_d_2)
print len(states_d_2)
print len(time_find_V_optimal_d_2)

plt.plot(  states_d_2, time_gather_v_bar_d_2, marker='o', label='gathering V vectors')
plt.plot(   states_d_2, time_find_V_optimal_d_2 ,marker='o',  linestyle='--', color='red', label='Finding optimal V vector')

plt.ylim(0,75)
plt.xlim(0,370)

plt.xlabel('number of states')
plt.ylabel('time(second)')
plt.title('time vs. number of states')
plt.legend(loc = 2)
plt.show()

#---------------------------------------------------------------
#na= 5, d =3. get average on 2 iterations.
# in designed algorithm, we extend nodes 10 times.
states_d_3 =  range(10,341,10)
queries_d_3 = [4.99, 5.05, 5.57, 5.82, 6.04, 5.61, 5.74, 6.21, 6.29, 5.81, 6.54, 6.44, 5.95, 6.17, 6.2, 6.42, 5.89, 5.96,
               5.81, 6.22, 5.96, 6.33, 6.33, 6.32, 6.19, 6.0, 6.51, 6.12, 6.26, 6.46, 6.26, 6.16, 6.19, 6.05, 5.76]
time_gather_v_bar_d_3 = [0.27524699688, 0.693392601013, 1.15753515244,1.67214765072, 2.21573509693,2.96884720087,
                         3.61708112717,4.32287112713, 4.97943257809, 5.77565413237,6.51969896793,  7.25336931229,
                         8.09471034288,  8.95989708662, 10.5371990895,  11.4606123352, 12.4454206181, 13.4594986176,
                         14.5079832149, 15.6311010265, 16.6167330241,18.0202090263, 19.2662728071,20.4055335021,
                         21.7099875998,  23.2792775011,25.0598808074, 26.7072791386, 29.3822305226, 30.1691702151,
                         31.5046087456, 33.5161254954, 35.3500837636,36.9195783019 ]

time_find_V_optimal_d_3 = [8.47523100138, 10.265344367, 11.6054734135,13.1665148282,  15.5932645512, 13.9659647679,
                           14.4796924543, 17.3719204473, 16.2973264337,  16.073357861,18.4332910299, 18.2790505171,
                           15.6813397408,17.7710701585, 29.3699855113, 34.550726335,  34.2905036736,30.8806801677,
                          29.7226494312,35.8062700868, 32.2896555734, 35.5931958699, 33.5171498728,  34.108456707,
                            33.078184135,17.3343445826, 15.4665324378,16.0489504457, 29.3822305226,  20.900325613,
                           16.5302208352,16.7780056643, 21.0202920747,25.673955636  ]
error_d_3 = [ 1.91301452294,  2.18717359422, 1.56344964601,  2.81221076312,  3.16836137751,3.19384489279, 2.9122098591,
             2.00654227297, 3.30424287341,  3.34067950186, 3.0994596127,2.38492813509, 3.05803547094,3.30315974845,
              2.0576834216,  2.70661586064, 1.69341147076,  2.00457825119, 2.56287051216,3.39310907324, 2.56365731605,
             2.72832840217, 3.07148891495, 2.92709202505,  1.67535208164,1.75634421522, 3.37968057083,2.13693439745,
              3.32633929107, 2.20368684283,1.78960277498, 3.46844772938,2.51978962873,2.62136772631 ]


plt.plot(states_d_3, time_gather_v_bar_d_3, marker='o', label='gathering V vectors')
plt.plot(states_d_3, time_find_V_optimal_d_3 ,marker='o',  linestyle='--', color='red', label='Finding optimal V vector')

#plt.ylim(0,75)
#plt.xlim(0,370)

plt.xlabel('number of states')
plt.ylabel('time(second)')
plt.title('time vs. number of states')
plt.legend(loc = 2)
plt.show()
#---------------------------------------------------------------
#na= 5, d =4. get average on 2 iterations.
# in designed algorithm, we extend nodes 10 times.
states_d_4 =range(10, 171, 10)
queries_d_4 = [10.56, 11.89, 12.34, 12.36, 12.05, 12.14, 12.14, 12.57, 12.51 ,12.19, 12.48, 12.74, 13.6, 12.68, 12.86,
               12.75, 12.39]
time_gather_v_bar_d_4 = [ 0.434765527248, 1.07288795471, 1.61044191837, 2.39948519707,  3.21667774677, 4.50905492544,
                          5.57666619062,6.76892741442,  8.02971940041,9.26669487476 ,  10.7363507318,12.113296001,
                          13.6907690644, 15.1012629771, 18.0503441381, 19.8201467371,  21.6137098742  ]

time_find_V_optimal_d_4 = [ 80.2153472972, 95.6004718041, 99.7748590517,  108.59150238,119.628235762, 159.217179105,
                           173.806535857, 182.284675455, 192.773511438,  188.449214711,  183.09602375,145.843103154,
                           212.88902199, 139.692160461, 151.22987993,140.35430444,  173.432496834 ]
error_d_4 = [ 2.22153439295, 2.91661503161,  2.35173014241, 2.34489754381,  2.34104276723,3.76851493529, 1.89947722793,
             3.87307262547, 3.30535403606,3.43789232786,2.84581458539, 3.41867777491, 2.43031455314, 3.56927187432,
             2.99675478434, 3.04994504218, 2.6591516614  ]


plt.plot(states_d_4, time_gather_v_bar_d_4, marker='o', label='gathering V vectors')
plt.plot(states_d_4, time_find_V_optimal_d_4 ,marker='o',  linestyle='--', color='red', label='Finding optimal V vector')

#plt.ylim(0,75)
#plt.xlim(0,370)

plt.xlabel('number of states')
plt.ylabel('time(second)')
plt.title('time vs. number of states')
plt.legend(loc = 2)
plt.show()




plt.plot(states_d_2, error_d_2, marker='o', label='d=2')
plt.plot(states_d_3, error_d_3, marker='o', linestyle='--', color='red',label='d=3')
plt.plot(states_d_4, error_d_4, marker='o', linestyle='-', color='green',label='d=4')

#plt.ylim(0,75)
#plt.xlim(0,370)

plt.xlabel('number of states')
plt.ylabel('error')
plt.title('error vs. number of states')
plt.legend(loc = 3)
plt.show()