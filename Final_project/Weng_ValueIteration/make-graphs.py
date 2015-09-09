from matplotlib import pyplot as plt

# #d= 2, s=128
# #weng with no noise
# queries_weng_d_2_128 =  [0, 12, 14, 15, 17, 18]
# errors_weng_d_2_128 = [8.4814750701189041, 8.027682900428772, 7.6162465810775757, 6.8639347553253174, 6.1866474151611328, 0.0033197402954101562]
# #weng with noise
# queries_weng_d_2_noise_128 = [0, 12, 14, 15, 17, 18]
# errors_weng_d_2_noise_128 = [8.4814750701189041, 8.1409767270088196, 7.8777084350585938, 7.6279431581497192, 2.8188309669494629]
# #advantages without noise
# queries_advantage_d_2_128 = [0, 1, 3, 6, 9, 11, 12, 13, 14, 15]
# errors_advantage_d_2_128 = [8.6942358016967773, 8.4814750701189041, 8.4524387717247009, 8.2451280951499939, 7.8194053769111633, 7.6089819669723511, 7.3823539018630981, 4.520627498626709, 0.98182821273803711, 0.0049061775207519531]
# #advantages with noise
# queries_advantage_d_2_noise_128 = [0, 1, 3, 5, 6]
# errors_advantage_d_2_noise_128 = [8.6942358016967773, 8.4814750701189041, 6.8435890674591064, 2.1998064517974854, 1.1781661510467529]


#d= 2, s=128
#weng with no noise
queries_weng_d_2_128 = [0, 17, 20, 21, 22, 23]
errors_weng_d_2_128 = [8.0430076718330383, 7.2250630855560303, 6.86536705493927, 4.9557693004608154, 4.2469320297241211,
                       0.024316310882568359]
#weng with noise
queries_weng_d_2_noise_128 = [0, 7, 10, 11, 12]
errors_weng_d_2_noise_128 = [8.0430076718330383, 7.1731411218643188, 5.6573753356933594, 3.9377322196960449, 1.0485391616821289]
#advantages without noise
queries_advantage_d_2_128 = [0, 1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15, 17]
errors_advantage_d_2_128 = [8.2726535797119141, 8.0430076718330383, 7.9801303744316101, 7.8113630712032318, 7.413032054901123,
                            7.0306718349456787, 6.6404346227645874, 5.3984532356262207, 2.6730303764343262, 2.4474420547485352,
                            2.2860369682312012, 2.0590806007385254, 0.48445940017700195]
#advantages without noise
queries_advantage_d_2_noise_128 = [0, 1, 3, 5, 7, 8, 9, 10]
errors_advantage_d_2_noise_128 = [8.2726535797119141, 8.0430076718330383, 7.9801303744316101, 7.7683311700820923,
                                  5.2385234832763672, 3.0955171585083008, 0.34446573257446289, 0.025012493133544922]


# #d= 2, s=256
# #weng with no noise
# queries_weng_d_2_256 = [0, 10, 11, 12, 13, 14]
# errors_weng_d_2_256 =  [10.640900671482086, 10.147556602954865, 7.8365190029144287, 7.4427754878997803, 5.4708251953125, 0.0035221576690673828]
# #weng with noise
# queries_weng_d_2_noise_256 =  [0, 10, 11, 12, 13, 14]
# errors_weng_d_2_noise_256 =  [10.640900671482086, 10.561661303043365, 10.460316598415375, 9.9353706836700439]
# #advantages without noise
# queries_advantage_d_2_256 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12]
# errors_advantage_d_2_256 = [10.89128303527832, 10.640900671482086, 10.614336162805557, 10.397982478141785, 9.7300810813903809, 8.1239683628082275, 4.8571853637695312, 1.55615234375, 0.20345854759216309, 0.13269615173339844]
# #advantages with  noise
# queries_advantage_d_2_noise_256 = [0, 1, 2, 3, 4, 5, 6]
# errors_advantage_d_2_noise_256 = [10.89128303527832, 10.640900671482086, 10.614336162805557, 9.7797145843505859, 8.9361345767974854, 3.9853358268737793, 0.28456306457519531]


#d= 2, s=256
#weng with no noise
queries_weng_d_2_256 =  [0, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25]
errors_weng_d_2_256 = [9.8225928843021393, 9.3579825758934021, 7.5504560470581055, 7.1548347473144531, 6.436920166015625,
                       4.7259845733642578, 4.4874777793884277, 4.0514860153198242, 3.4708256721496582, 3.2970428466796875,
                       0.0039997100830078125]
#weng with noise
queries_weng_d_2_noise_256 = [0, 4]
errors_weng_d_2_noise_256 =  [9.8225928843021393, 2.1266720294952393]
#advantages without noise
queries_advantage_d_2_256 =  [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
errors_advantage_d_2_256 =  [10.06462574005127, 9.8225928843021393, 9.7928587794303894, 9.5635733604431152,
                             8.8444585800170898, 8.1287064552307129, 7.9029827117919922, 7.0326929092407227,
                             6.1690528392791748, 4.3078527450561523, 3.7680721282958984, 3.2793669700622559,
                             1.7361593246459961, 1.3434622287750244, 1.1396386623382568, 0.7763824462890625,
                             0.067195892333984375]
#advantages without noise
queries_advantage_d_2_noise_256 = [0, 1, 3, 4, 5, 6, 7, 8]
errors_advantage_d_2_noise_256 = [10.06462574005127, 10.06462574005127, 9.7928587794303894, 8.5068469047546387,
                                  8.4247153997421265, 8.5476394891738892, 8.8838872909545898, 8.8628072738647461]





# #--------------------------------------------------------------------------------------
# #d= 3, s=128
# #weng with no noise
# queries_weng_d_3_128 =   [0, 24, 29, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
# errors_weng_d_3_128 =  [7.9955092668533325, 7.6668994426727295, 7.3016650080680847, 6.9395773410797119, 6.6040472984313965,
#         6.2879762649536133, 5.6893069744110107, 5.4081709384918213, 4.6454851627349854, 4.1961452960968018, 2.9326534271240234, 2.6467685699462891, 2.5144233703613281, 2.2692046165466309, 2.1557011604309082, 1.9454159736633301, 1.8480844497680664, 1.6677732467651367, 1.5843167304992676, 1.2901854515075684, 0.62847423553466797, 0.017847061157226562, 0.0066461563110351562, 0.0074028968811035156]
# #weng with noise
# queries_weng_d_3_noise_128 =  [0, 24, 29, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
# errors_weng_d_3_noise_128 =   [7.9955092668533325, 6.4936081171035767, 5.8207190036773682, 5.5041446685791016, 4.6334602832794189, 1.8717267513275146]
# #advantages without noise
# queries_advantage_d_3_128 =   [0, 2, 3, 11, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35, 36, 37]
# errors_advantage_d_3_128 = [8.1549644470214844, 7.9955092668533325, 7.9210690408945084, 7.7581069469451904, 7.546278178691864,
#         7.1086345911026001, 5.2540271282196045, 2.9158229827880859, 2.4436545372009277, 2.3473472595214844, 1.5986332893371582, 0.79850149154663086, 0.76496553421020508, 0.69857120513916016, 0.59520602226257324, 0.57308375835418701, 0.56890690326690674, 0.56532740592956543, 0.4651484489440918, 0.42355608940124512, 0.35385227203369141, 0.32232952117919922, 0.091296195983886719, 0.051771163940429688, 0.033302068710327148]
# #advantages with noise
# queries_advantage_d_3_noise_128  = [0, 2, 6, 8, 10, 13, 14, 15, 16, 17, 18]
# errors_advantage_d_3_noise_128 = [8.1549644470214844, 8.1549644470214844, 7.8854669332504272, 7.7787171006202698, 7.6448630094528198,
#         7.2598729729652405, 5.534621000289917, 5.4615027904510498, 5.4182446002960205, 5.2735855579376221, 5.9823222160339355]


#d= 3, s=128
#weng with no noise
queries_weng_d_3_128 = [0, 23, 26, 27, 29, 30, 31, 32]
errors_weng_d_3_128 = [8.7374642640352249, 8.3221808671951294, 7.9201544523239136, 6.750185489654541, 5.7897722721099854,
                       5.5041089057922363, 5.2271466255187988, 0.067395210266113281]
#weng with noise
queries_weng_d_3_noise_128 = [0, 8, 10]
errors_weng_d_3_noise_128 = [8.7374642640352249, 8.4599116146564484, 9.0971250534057617]
#advantages without noise
queries_advantage_d_3_128 = [0, 2, 7, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33]
errors_advantage_d_3_128 = [8.9125576019287109, 8.7374642640352249, 8.6631007790565491, 8.3212922811508179,
                            7.3751797676086426, 7.164318323135376, 6.5198476314544678, 5.5071146488189697,
                            4.8428750038146973, 3.8860063552856445, 3.7607212066650391, 3.4610161781311035,
                            3.0675787925720215, 1.7805581092834473, 0.73065924644470215, 0.6945350170135498,
                            0.061084508895874023, 0.053279638290405273, 0.25225663185119629]
#advantages with noise
queries_advantage_d_3_noise_128 = [0, 2, 3, 7, 9, 10, 11, 12, 13, 14]
errors_advantage_d_3_noise_128 = [8.9125576019287109, 8.9125576019287109, 8.735377699136734, 8.6239649951457977,
                                  8.3799900412559509, 7.7659981250762939, 7.2497889995574951, 6.7421984672546387,
                                  8.0770063400268555, 8.0638665556907654]


# #d= 3, s=256
# #weng with no noise
# queries_weng_d_3_256 = [0, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]
# errors_weng_d_3_256 = [9.0137824565172195, 8.6787618398666382, 8.2773601412773132, 7.8625805377960205, 7.4619784355163574,
#         7.102414608001709, 6.7427191734313965, 6.3980679512023926, 5.76407790184021, 4.2365789413452148, 4.0247573852539062, 0.0068384408950805664]
# #weng with noise
# queries_weng_d_3_noise_256 = [0, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]
# errors_weng_d_3_noise_256 = [9.0137824565172195, 8.8723903298377991, 8.6516767740249634, 8.5820199251174927, 7.3815816640853882]
# #advantages without noise
# queries_advantage_d_3_256 = [0, 2, 8, 18, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
# errors_advantage_d_3_256 = [9.1773090362548828, 9.0137824565172195, 8.9459388107061386, 8.5793139934539795, 8.3673796653747559,
#         7.9260380268096924, 6.7520365715026855, 5.1870880126953125, 5.0529637336730957, 4.8079485893249512, 3.3741302490234375,
#         2.5061445236206055, 1.6075630187988281, 1.474698543548584, 1.0848616361618042, 1.0880032777786255, 1.0046991109848022,
#         0.95305502414703369, 0.55782878398895264, 0.031021595001220703, 0.029134869575500488, 0.015138506889343262, 0.0039101839065551758]
# #advantages without noise
# queries_advantage_d_3_noise_256 =  [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# errors_advantage_d_3_noise_256 = [9.1773090362548828, 9.1773090362548828, 9.0132056772708893, 8.5300904512405396, 7.9688408374786377,
#         7.6657719612121582, 7.0313358306884766, 6.7802994251251221, 7.0081744194030762, 7.4123800992965698, 7.34548020362854, 7.4108150005340576, 7.4135222434997559]



#d= 3, s=256
#weng with no noise
queries_weng_d_3_256 = [0, 23, 24, 28, 29, 30]
errors_weng_d_3_256 = [8.2299897521734238, 7.8749673962593079, 7.4642487168312073, 4.9426712989807129, 4.4600605964660645,
                       0.0046579837799072266]
#weng with noise
queries_weng_d_3_noise_256 = [0, 18, 19]
errors_weng_d_3_noise_256 = [8.2299897521734238, 7.339847207069397, 5.8391404151916504]
#advantages without noise
queries_advantage_d_3_256 = [0, 2, 3, 10, 15, 17, 18, 20, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
errors_advantage_d_3_256 = [8.39703369140625, 8.2299897521734238, 8.1666299104690552, 8.0179175138473511, 7.8687882423400879,
                            7.7015546560287476, 7.5404454469680786, 7.2434046268463135, 7.105241060256958, 6.621979832649231,
                            6.5139694213867188, 5.6784036159515381, 5.0685536861419678, 5.0042855739593506, 2.4207196235656738,
                            1.6320619583129883, 1.3681678771972656, 1.3153657913208008, 1.3054215908050537, 1.2539165019989014,
                            1.1450371742248535, 0.70594048500061035, 0.69871902465820312, 0.63726711273193359, 0.44658946990966797,
                            0.41765928268432617, 0.075954675674438477]
#advantages without noise
queries_advantage_d_3_noise_256 = [0, 2, 3, 6, 7, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21]
errors_advantage_d_3_noise_256 = [8.39703369140625, 8.2299897521734238, 8.1666299104690552, 8.0219844281673431,
                                  7.8462296724319458, 7.1095691919326782, 6.1890749931335449, 5.403656005859375,
                                  3.9756245613098145, 2.1584105491638184, 1.6623320579528809, 1.4177365303039551,
                                  0.8769460916519165, 0.88313829898834229, 0.85961854457855225]


#--------------------------------------------------------------------------------------
#d= 4, s=128
#weng with no noise
queries_weng_d_4_128 = [0, 34, 37, 40, 42, 45, 46, 47, 48, 49, 50]
errors_weng_d_4_128 = [6.8859360218048096, 6.6160575151443481, 6.2508364915847778, 5.9272991418838501, 5.6523504257202148,
        4.8737149238586426, 4.6379463672637939, 3.2426981925964355, 2.7792239189147949, 2.639894962310791, 0.020956516265869141]
#weng with noise
queries_weng_d_4_noise_128 = [0, 34, 37, 40, 42, 45, 46, 47, 48, 49, 50]
errors_weng_d_4_noise_128 =  [6.8859360218048096, 6.2269690036773682, 5.5484819412231445, 4.1020185947418213,
                          3.3849315643310547, 2.1257911920547485]
#advantages without noise
queries_advantage_d_4_128 = [0, 3, 10, 12, 14, 20, 24, 26, 30, 32, 37, 38, 39, 41, 42, 43, 44, 45, 47, 49, 51, 52, 53, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67]
errors_advantage_d_4_128 = [7.0131301879882812, 6.8859360218048096, 6.7957256436347961, 6.6660493910312653, 6.4866375923156738,
                        6.2802258133888245,5.8758982419967651, 5.6617738008499146, 5.4423956871032715, 5.2490301132202148,
                        5.0503302812576294, 4.8772189617156982, 4.721529483795166, 4.5635936260223389, 4.2809615135192871,
                        3.900439977645874, 3.5695147514343262, 3.2109832763671875, 3.0449244976043701, 2.2109436988830566,
                        2.1453189849853516, 2.0146994590759277, 1.9860152006149292, 1.9748001098632812, 1.95451819896698,
                        1.8033688068389893, 1.7946925163269043, 1.5120984315872192, 1.4744448661804199, 1.4463413953781128,
                        1.3836034536361694, 1.3676073551177979, 1.30597984790802, 1.0892000198364258, 0.021403789520263672]
#advantages without noise
queries_advantage_d_4_noise_128 = [0, 3, 8, 10, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
errors_advantage_d_4_noise_128 = [7.0131301879882812, 6.8859360218048096, 6.6912173330783844, 6.5632366538047791,
                              6.2344774007797241, 6.0779522657394409,5.3655083179473877, 5.0104398727416992,
                              4.8067550659179688, 4.7139697074890137, 4.4637401103973389, 4.1768009662628174,
                              3.3492567539215088, 3.3272311687469482, 3.0455105304718018, 2.0759015083312988,
                              2.0819249153137207, 3.0617506504058838, 3.949631929397583]

#d= 4, s=256
#weng with no noise
queries_weng_d_4_256 = [0, 37, 46, 50, 53, 58, 63, 67, 69, 70, 71, 73, 74, 75, 76, 77]
errors_weng_d_4_256 = [7.2268183082342148, 6.9776060879230499, 6.7527663111686707, 6.4812841415405273, 6.2456690073013306,
                 5.9951184988021851, 5.7109982967376709, 5.4657021760940552, 5.0238068103790283, 4.5602350234985352,
                 4.3312158584594727, 3.8994710445404053, 3.7163217067718506, 3.529930591583252, 3.3565165996551514,
                 0.0096815824508666992]
#weng with noise
queries_weng_d_4_noise_256 = [0, 37, 46, 50, 53, 58, 63, 67, 69, 70, 71, 73, 74, 75, 76, 77]
errors_weng_d_4_noise_256 = [7.2268183082342148, 7.0652052760124207, 6.984819233417511, 6.6987893581390381,
                         6.6213722825050354, 8.13139408826828]
#advantages without noise
queries_advantage_d_4_256 = [0, 3, 14, 20, 28, 33, 35, 38, 39, 41, 42, 45, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                         61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                         85, 86, 87, 88, 89, 90]
errors_advantage_d_4_256 = [7.3563008308410645, 7.3563008308410645, 7.1306338012218475, 6.9891681373119354, 6.8041712045669556,
                        6.614721417427063, 6.4368734359741211, 6.2500882148742676, 6.0843894481658936, 5.7520848512649536,
                        5.5939394235610962, 5.447580099105835, 5.318706750869751, 4.9578280448913574, 4.0087966918945312,
                        3.9587521553039551, 3.0425195693969727, 2.5328460931777954, 2.5239509344100952, 2.5210249423980713,
                        2.51244056224823, 2.4510481357574463, 2.4436521530151367, 2.4387075901031494, 2.4392353296279907,
                        2.4294229745864868, 2.4280874729156494, 2.1330578327178955, 1.9805183410644531, 1.8885297775268555,
                        0.97373795509338379, 0.79004406929016113, 0.65242934226989746, 0.54816818237304688, 0.51962566375732422,
                        0.29295921325683594, 0.34219551086425781, 0.36462211608886719, 0.36770009994506836, 0.38960838317871094,
                        0.36080026626586914, 0.39472627639770508, 0.45022916793823242, 0.50745129585266113, 0.51059079170227051,
                        0.51729583740234375, 0.518310546875, 0.52109313011169434, 0.43505477905273438, 0.41826009750366211,
                        0.39806127548217773, 0.39795398712158203, 0.39785432815551758]




#d= 5, s=128
#weng with no noise
queries_weng_d_5_128 = [0, 51, 57, 60, 69, 72, 76, 78, 79, 80, 81, 83]
errors_weng_d_5_128 = [5.2959152907133102, 5.0967766046524048, 4.8587335348129272, 4.6181560754776001, 4.3947362303733826,
                   3.9098916053771973, 3.4982539415359497, 3.3048923015594482, 3.1263542175292969, 2.9728243350982666,
                   2.6656103134155273, 0.048036098480224609]
#weng with noise
queries_weng_d_5_noise_128 = [0, 51, 57, 60, 69, 72, 76, 78, 79, 80, 81, 83]
errors_weng_d_5_noise_128 = [5.2959152907133102, 6.3388078808784485]
#advantages without noise
queries_advantage_d_5_128 = [0, 4, 17, 26, 33, 38, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52, 54, 57, 58, 59, 60, 61, 63,
                         64, 65, 66, 67, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 86, 87, 88, 89, 90, 91,
                         92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106]
errors_advantage_d_5_128 = [5.3757033348083496, 5.3757033348083496, 5.1933610141277313, 5.1031177639961243, 4.816689670085907,
                        4.6802488565444946, 4.4163832068443298, 4.2939367294311523, 4.1830872297286987, 3.8663390874862671,
                        3.6813180446624756, 3.3798394203186035, 3.0266838073730469, 2.9761850833892822, 2.8858094215393066,
                        2.7647299766540527, 2.4720790386199951, 2.3720705509185791, 2.248016357421875, 2.2249588966369629,
                        2.3086843490600586, 2.3057551383972168, 2.267244815826416, 2.25221848487854, 2.2438344955444336,
                        2.2091538906097412, 2.1789813041687012, 2.1534092426300049, 2.0498595237731934, 2.0417602062225342,
                        1.9438660144805908, 1.4756894111633301, 1.4449582099914551, 1.149799108505249, 1.1458292007446289,
                        1.1079432964324951, 0.95160746574401855, 0.94593024253845215, 0.9397575855255127, 0.93149018287658691,
                        0.90505075454711914, 0.90209293365478516, 0.90151095390319824, 0.9056398868560791, 1.0122616291046143,
                        1.2092119455337524, 1.2239208221435547, 1.2300214767456055, 1.2495801448822021, 1.3047147989273071,
                        1.3122868537902832, 1.3070641756057739, 1.3031033277511597, 1.2893288135528564, 1.1996771097183228,
                        0.82796168327331543, 0.76871371269226074, 0.4330742359161377, 0.42204737663269043, 0.20856857299804688,
                        0.15281581878662109, 0.020268440246582031, 0.04431915283203125]
#advantages without noise
queries_advantage_d_5_noise_128 = [0, 4, 9, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
errors_advantage_d_5_noise_128 = [5.3757033348083496, 5.3757033348083496, 5.1933610141277313, 4.9598435163497925,
                              4.4571161866188049, 3.9916231632232666, 2.9707658290863037, 2.8252809047698975,
                              2.2323095798492432, 2.194981575012207, 1.7616808414459229, 1.5726600885391235,
                              1.4824056625366211, 1.5006237030029297, 1.3944485187530518, 1.6790046691894531,
                              1.6987737417221069]

#d= 5, s=256
#weng with no noise
queries_weng_d_5_256 = [0, 61, 70, 75, 77, 78, 81, 82, 83, 85, 87, 89, 93, 95]
errors_weng_d_5_256 = [5.8783963844180107, 5.696486234664917, 5.457750678062439, 5.1540131568908691, 4.6163846254348755,
                   4.1447616815567017, 3.9320321083068848, 3.7294437885284424, 3.5371770858764648, 3.3639123439788818,
                   3.1978302001953125, 3.0281069278717041, 2.0887582302093506, 0.0033578872680664062]
#weng with noise
queries_weng_d_5_noise_256 = [0, 61, 70, 75, 77, 78, 81, 82, 83, 85, 87, 89, 93, 95]
errors_weng_d_5_noise_256 = [5.8783963844180107, 5.6639485359191895, 5.0604579448699951, 4.4879947900772095,
                         4.2360366582870483, 3.5335311889648438, 3.3207862377166748, 2.8549633026123047,
                         2.789547324180603, 2.7301849126815796, 2.1792000532150269]



plt.plot( queries_weng_d_2_128, errors_weng_d_2_128,marker='o', label='IVI d=2')
plt.plot( queries_advantage_d_2_128, errors_advantage_d_2_128, marker='o', linestyle='--', color='black', label='ABVI d=2')

# plt.plot( queries_weng_d_3_128, errors_weng_d_3_128, marker='s', label='IVI d=3')
# plt.plot( queries_advantage_d_3_128, errors_advantage_d_3_128, marker='o', linestyle='--', label='ABVI d=3')

plt.plot( queries_weng_d_4_128, errors_weng_d_4_128, marker='+', label='IVI d=4')
plt.plot( queries_advantage_d_4_128, errors_advantage_d_4_128, linestyle='--', marker='+', label='ABVI d=4')

# plt.plot( queries_weng_d_5_128, errors_weng_d_5_128, marker='s', label='IVI d=5')
# plt.plot( queries_advantage_d_5_128, errors_advantage_d_5_128, marker='o', linestyle='--', label='ABVI d=5')

# plt.ylim(0,10)
# plt.xlim(0,200)

plt.xlabel('number of queries')
plt.ylabel('error')
plt.title('error vs. number of queries')
plt.legend()
plt.show()





#plt.plot( queries_weng_d_2_128, errors_weng_d_2_128, marker='s', label='IVI d=2')
#plt.plot( queries_advantage_d_2_128, errors_advantage_d_2_128, marker='o', linestyle='--', label='ABVI d=2')

plt.plot( queries_weng_d_3_128, errors_weng_d_3_128, marker='o', label='IVI d=3')
plt.plot( queries_advantage_d_3_128, errors_advantage_d_3_128, marker='o', linestyle='--', label='ABVI d=3')

#plt.plot( queries_weng_d_4_128, errors_weng_d_4_128, marker='s', label='IVI d=4')
#plt.plot( queries_advantage_d_4_128, errors_advantage_d_4_128, marker='o', linestyle='--', label='ABVI d=4')

plt.plot( queries_weng_d_5_128, errors_weng_d_5_128,marker='+', label='IVI d=5')
plt.plot( queries_advantage_d_5_128, errors_advantage_d_5_128, linestyle='--',marker='+', label='ABVI d=5')

# plt.ylim(0,10)
# plt.xlim(0,200)

plt.xlabel('number of queries')
plt.ylabel('error')
plt.title('error vs. number of queries')
plt.legend()
plt.show()




# #plt.plot( queries_weng_d_2_256, errors_weng_d_2_256, marker='s', label='IVI d=2')
# #plt.plot( queries_advantage_d_2_256, errors_advantage_d_2_256, marker='o', linestyle='--', label='ABVI d=2')
#
# plt.plot( queries_weng_d_3_256, errors_weng_d_3_256, marker='o', label='IVI d=3')
# plt.plot( queries_advantage_d_3_256, errors_advantage_d_3_256, marker='s', linestyle=':', label='ABVI d=3')
#
# #plt.plot( queries_weng_d_4_256, errors_weng_d_4_256, marker='s', label='IVI d=4')
# #plt.plot( queries_advantage_d_4_256, errors_advantage_d_4_256, marker='o', linestyle='--', label='ABVI d=4')
#
# plt.plot( queries_weng_d_5_256, errors_weng_d_5_256, linestyle='--',marker='*', label='IVI d=5')
# plt.plot( queries_advantage_d_5_256, errors_advantage_d_5_256, linestyle='-.',marker='+', label='ABVI d=5')
#
# # plt.ylim(0,10)
# # plt.xlim(0,200)
#
# plt.xlabel('number of queries')
# plt.ylabel('error')
# plt.title('error vs. number of queries')
# plt.legend()
# plt.show()


plt.plot( queries_weng_d_2_256, errors_weng_d_2_256, marker='o', label='IVI d=2')
plt.plot( queries_advantage_d_2_256, errors_advantage_d_2_256, marker='o', linestyle='--', label='ABVI d=2')

plt.plot( queries_weng_d_3_256, errors_weng_d_3_256, marker='<', label='IVI d=3')
plt.plot( queries_advantage_d_3_256, errors_advantage_d_3_256, marker='<', linestyle='--', label='ABVI d=3')

plt.plot( queries_weng_d_4_256, errors_weng_d_4_256, marker='+', label='IVI d=4')
plt.plot( queries_advantage_d_4_256, errors_advantage_d_4_256, marker='+', linestyle='--', label='ABVI d=4')

# plt.plot( queries_weng_d_5_256, errors_weng_d_5_256, linestyle='--',marker='*', label='IVI d=5')
# plt.plot( queries_advantage_d_5_256, errors_advantage_d_5_256, linestyle='-.',marker='+', label='ABVI d=5')

plt.xlim(0,95)
# plt.xlim(0,200)

plt.xlabel('number of queries')
plt.ylabel('error')
plt.title('error vs. number of queries')
plt.legend()
plt.show()




plt.plot( queries_weng_d_2_noise_128, errors_weng_d_2_noise_128, marker='o', label='IVI d=2')
plt.plot( queries_advantage_d_2_noise_128, errors_advantage_d_2_noise_128, marker='o', linestyle='--', label='ABVI d=2')

plt.plot( queries_weng_d_3_noise_128, errors_weng_d_3_noise_128, marker='<', label='IVI d=3')
plt.plot( queries_advantage_d_3_noise_128, errors_advantage_d_3_noise_128, marker='<', linestyle='--', label='ABVI d=3')

#plt.plot( queries_weng_d_4_noise_256, errors_weng_d_4_noise_256, marker='+', label='IVI d=4')
#plt.plot( queries_advantage_d_4_noise_256, errors_advantage_d_4_noise_256, marker='+', linestyle='--', label='ABVI d=4')

# plt.plot( queries_weng_d_5_noise_256, errors_weng_d_5_noise_256, linestyle='--',marker='*', label='IVI d=5')
# plt.plot( queries_advantage_d_5_noise_256, errors_advantage_d_5_noise_256, linestyle='-.',marker='+', label='ABVI d=5')

# plt.ylim(0,10)
plt.xlim(0,15)

plt.xlabel('number of queries')
plt.ylabel('error')
plt.title('error vs. number of queries')
plt.legend(loc = 3)
plt.show()