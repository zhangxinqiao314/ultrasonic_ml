import numpy as np
from matplotlib import pyplot as plt
import pickleJar as pj
import tomography as tm
from scipy.signal import butter, sosfiltfilt
import copy
import math
import time
from tqdm import tqdm
#

dir = 'C://Users//shams//Drexel University//Chang Lab - General//Individual//Sam Amsterdam//acoustic scan data//tomography_testing//example_data//'

# dirctionary with water by varying frequency

#
## keys of the data dict
# fileName
# 0:
# --voltage_transmission_forward
# --voltage_echo_forward
# --voltage_transmission_reverse
# --voltage_echo_reverse
# --voltageOffsetForward
# --gainForward
# --voltageOffsetReverse
# --gainReverse
# --time
# --time_collected
# --collection_index
# parameters:
# --gui
# --experiment
# --axis
# --distance
# --transducerFrequency
# --collectionMode
# --pulserType
# --measureTime
# --measureDelay
# --voltageRange
# --autoRange
# --autoRangeEcho
# --waves
# --samples
# --halfCycles
# --multiplexer
# --collectionDirection
# --voltageOffsetForward
# --voltageOffsetReverse
# --gainForward
# --gainReverse
# --picoModule
# --pulseModule
# --rfSwitch
# --t0PulseSwitch
# --t0ReceiveSwitch
# --t1PulseSwitch
# --t1ReceiveSwitch
# --experimentFolder
# --experimentName
# --experimentBaseName
# --saveFormat
# --postAnalysis
# --primaryAxis
# --secondaryAxis
# --primaryAxisRange
# --primaryAxisStep
# --secondaryAxisRange
# --secondaryAxisStep
# --scanInterval
# --numberOfScans
# --pulseInterval
# --experimentTime
# --pulserPort
# --scannerPort
# --dllFile
# --multiplexerPort
# --transducerHolderHeight
# --scannerMaxDimensions
# --fileName

waterFile = dir + 'SA_tomography_test_water.pickle'
waterDat = pj.loadPickle(waterFile)
water = waterDat[0]['voltage_transmission_forward']
waterechor = waterDat[0]['voltage_echo_reverse']

# butterworth filter for highpass filter TODO: where did you learn this? was originally part of tutorial, or from reading?
sos = butter(5, 1000000, btype = 'highpass', analog = False, fs = 500000000, output = 'sos')
refGain = waterDat[0]['gainReverse'] / 10

# change signal to pre-amplification voltage
refUngained = pj.correctVoltageByGain(waterechor, refGain)
# apply butterworth filter forward and reverse to pre-amplified signal
refFil = sosfiltfilt(sos, refUngained)

# trims 5 zero crossings around the first break, at 14100 index TODO: why these values?
ref = tm.trimByZeros(refFil, 14100, 5)
refX = np.linspace(0, (len(ref)-1) * 2, len(ref) - 1)

# load real data
# metalFile = dir + 'SA_tomography_test_metal_plate.pickle'
# metalDat = pj.loadPickle(metalFile)
# metalechor = metalDat[0]['voltage_echo_reverse']
# metalGain = metalDat[0]['gainReverse'] / 10
# metalUngained = pj.correctVoltageByGain(metalechor, metalGain)
# metalFil = sosfiltfilt(sos, metalUngained)

# generate testing data
# layer, dir, amp, time TODO: what is dir, direction?
tstWave = [0,1,3,0]
# z, loss, time
# test system: 20 mm H2O, 10 mm PLA (polylactic acid), 20 mm H2O
# H2O: 1.5 MRayl, 1500 m/s -> TT= 13,333 ns TODO: what is TT? Travel time?
# Rayl: specific acoustic impedance = pressure / particle velocity phasor
# PLA: density = 1.24 g/cm^3 c = 2,200 m/s -> TT = 4,545 ns
#   Z = 2.728 MRayl
dPLA = 1 # thickness in mm
ttPLA = dPLA * (1000 / 2.2) # travel time in ns, using 2200 m/s as speed of sound # TODO: will we assume all the c is known for materials?
dH2O = 25
ttH2O = dH2O * (1000 / 1.5)
# List of layers, in format[(impedance, thickness, travel time), ....]
# tstModel = [(1.5, 1, ttH2O), (2.7, 1, ttPLA), (1.5, 1, ttH2O)]
tstModel = [(1.5, 1, 2500), (2.7, 1, 1000), (1.5, 1, 2500)]
# tstCoeffs = tm.calculateReflectionAndTransmissionCoeffs(tstModel, 2)
# print(tm.propagateWaveThroughLayer(tstWave, tstLayer, 0.5, -0.2))
# print(tm.propagateWavesThroughLayers(tstWave, tstModel, tstCoeffs, -1, 0.01, 10))
# TODO: walk through the functions in tomography.py to understand 1) how the signal is generated, 2) which parameters we are trying to optimize for
signals = tm.generateSignalFromModel(ref, refX, tstModel, 'echo', 'reverse',
                                     3, 0.01, 20000, -1, False)
transSignals = tm.generateSignalFromModel(ref, refX, tstModel, 'transmission', 'reverse',
                                     3, 0.01, 20000, -1, False)

testSig = signals['echo_reverse']
testTime = signals['echo_reverse_time']
transSig = transSignals['transmission_reverse']
transTime = transSignals['transmission_reverse_time']
refX50mhz = np.linspace(0, (len(ref)-1) * 0.09, len(ref) - 1)
signals50mhz = tm.generateSignalFromModel(ref, refX50mhz, tstModel, 'echo', 'reverse',
                                     3, 0.01, 20000, -1, False)
test50mhz = signals50mhz['echo_reverse']
test50mhzTime = signals50mhz['echo_reverse_time']
plt.plot(testTime, testSig)
# plt.plot(test50mhzTime, test50mhz)
# plt.plot(transTime, transSig)
plt.show()
# test leading edge fit
# leadLen = 20
# shift, amp, mse = tm.leadingEdgeFit(testSig, ref, 0.01, leadLen)
# fit = tm.plotFits(ref, testSig, [shift], [amp])
#
# newSignal = testSig - fit
# newShift, newAmp, newMSE = tm.leadingEdgeFit(newSignal, ref, 0.01, leadLen)
# tm.plotFits(ref, newSignal, [newShift], [newAmp])
# tm.plotFits(ref, testSig, [shift, newShift], [amp, newAmp])



# # plt.plot(metalFil)
# plt.plot(testSig)
# plt.plot(ref)
# plt.show()


# refLead = ref[:50]
# refLen = len(ref)
# refDiff = len(ref) - len(refLead)
# refX = np.linspace(0, len(ref), len(ref))
# refToF = pj.envelopeThresholdTOF(ref, refX, 0.1) # this is used to calibrate how much of the signal start is missed by the tof algorithm
# signalLen = 3 * refLen
# signalX = np.linspace(0, signalLen, signalLen)
#
# # plt.plot(refUngained)
# # plt.plot(ref)
# # plt.show()
# testShifts = [refLen + 100, refLen + 300]
# testAmps = [1, 0.5]
# # weights = 10 * np.exp(-refX / 100)
# leadLen = 400
# neighborWidth = 10
# weights = np.concatenate((np.ones(leadLen), np.zeros(refLen - leadLen)))
# testSig = tm.generateSignalFromShiftAmp(ref, signalLen, testShifts, testAmps)
#
# # leading decomposition
# currentSignal = copy.copy(testSig)
# amps = []
# shifts = []
# stepMSEs = []
# mses = []
# cumRes = []
# mseThreshold = 0.01
# maxIter = 100
# iter = 0
# stopFitting = False
#
# #todo: its working if ref lead is perfectly specified. Need to make things more robust to noise - better tof algorithm with higher noise tolerance
# while not stopFitting:
#
#     # calculate tof of current signal, use that to calculate a shift
#     tof = pj.envelopeThresholdTOF(currentSignal, signalX, 0.1)
#     # adjust by the ref ToF and add refLen since shift is based on the right side of the wave
#     shiftGuess = math.floor(tof - refToF + refLen)
#
#     # calculate a heavily front weighted fit at the shift
#     fits = tm.weightedCorrelationFit(currentSignal, ref, weights)
#
#     # find the best fit near the shift guess
#     indexRange = range(shiftGuess - neighborWidth, shiftGuess + neighborWidth)
#     minMSE = np.min(fits[1][indexRange])
#     minInd = np.argwhere(fits[1][indexRange] == minMSE)[0][0] + shiftGuess - neighborWidth + 1
#
#     # grab the fit at the nearest index to shift
#     #todo: linear interpolation
#     amp = fits[0][minInd]
#     stepMSE = fits[1][minInd]
#
#     # update fitting parameters
#     shifts.append(minInd)
#     amps.append(amp)
#     stepMSEs.append(stepMSE)
#
#     # plot
#     tm.plotFits(ref, currentSignal, [minInd], [amp])
#     totalFit = tm.plotFits(ref, testSig, shifts, amps)
#     oldSignal = copy.copy(currentSignal)
#     currentSignal = oldSignal - totalFit
#
#     # calculate total mse
#     mse = np.sum((testSig - totalFit)**2) / signalLen
#     mses.append(mse)
#
#     iter += 1
#     # check whether to step
#     if iter > maxIter or mse < mseThreshold:
#         stopFitting = True
#
#
# print(shifts)
# print(amps)
# print(mses)
#
# #########################################################################
# ########## Testing ##############################################
# #########################################################
# # # run random tests
# # numberOfTrials = 10
# # minRefs = 2
# # maxRefs = 10
# #
# # testCases = []
# # numErrs = []
# # ampErrs = []
# # shiftErrs = []
# # ress = []
# #
# # for i in tqdm(range(numberOfTrials)):
# #
# #     # generate random decomposition
# #     numRefs = np.random.randint(minRefs, maxRefs)
# #     shifts = np.random.rand(numRefs) * signalLen
# #     amps = np.random.rand(numRefs)
# #     testCases.append([numRefs, shifts, amps])
# #
# #     # sort by amplitudes so that they can be compared with the decomposition
# #     ampShifts = np.array([amps, shifts]).T
# #     sortedAmpShifts = sorted(ampShifts, key = lambda x: x[0])
# #
# #     # generate signal and decompose
# #     signal = tm.generateSignalFromShiftAmp(ref, signalLen, shifts, amps)
# #     decomp = tm.matchingPursuitDecomposition(ref, signal, normResThreshold = 0.05, plotResult = False)
# #     decompTranspose = np.array([decomp[1], decomp[0]]).T
# #     sortedDecomp = sorted(decompTranspose, key = lambda x: x[0])
# #
# #     # calculate errors
# #     numberErr = len(decomp[0]) - numRefs
# #     numErrs.append(numberErr)
# #     # calculate element-wise error if there are equal or less decomps to actual
# #     # for unequal numbers of elements, make the comparison for as many elements as possible
# #     compareLen = min(len(decomp[0]), numRefs)
# #     ampErr = [abs(sortedDecomp[i][0] - sortedAmpShifts[i][0]) for i in range(compareLen)]
# #     ampErrs.append(ampErr)
# #     shiftErr = [abs(sortedDecomp[i][1] - sortedAmpShifts[i][1]) for i in range(compareLen)]
# #     shiftErrs.append(shiftErr)
# #     res = decomp[2][-1]
# #     ress.append(res)
# #
# # def safeMax(input):
# #     try:
# #         return max(input)
# #     except TypeError:
# #         return input
# #
# # maxAmpErrs = [safeMax(ampErrs[i]) for i in range(len(ampErrs))]
# # maxShiftErrs = [safeMax(shiftErrs[i]) for i in range(len(shiftErrs))]
# # plt.plot(range(numberOfTrials), ress)
# # plt.plot(range(numberOfTrials), numErrs)
# # plt.plot(range(numberOfTrials), maxAmpErrs)
# # plt.plot(range(numberOfTrials), maxShiftErrs)
# # plt.show()
# #
# # # tstShifts = [100 + refLen, 800 + refLen]
# # # tstAmps = [1, 0.5]
# # # tstSum = tm.generateSignalFromShiftAmp(ref, signalLen, tstShifts, tstAmps)
# # # tstDecomp = tm.matchingPursuitDecomposition(ref, tstSum, normResThreshold = 0.05, plotSteps = True)
# # # tm.plotFits(ref, tstSum, tstShifts, tstAmps)
# # # print(tstDecomp)
# #
# # # plt.plot(ref)
# # # plt.plot(tstSum)
# # # plt.show()

# testing tomography model output
# get ref info
# refTime = waterDat[0]['time']
# refIndices = range(13700, 15120)
# ref = refFil[refIndices]
# refX = refTime[refIndices]
# # plt.plot(refTime[refIndices], refFil[refIndices])
# # plt.show()
#
# # layer, dir, amp, time
# tstWave = [0,1,3,0]
# # z, loss, time
# # test system: 20 mm H2O, 10 mm PLA, 20 mm H2O
# # H2O: 1.5 MRayl, 1500 m/s -> TT= 13,333 ns
# # PLA: density = 1.24 g/cm^3 c = 2,200 m/s -> TT = 4,545 ns
# #   Z = 2.728 MRayl
# dPLA = 1 # thickness in mm
# ttPLA = dPLA * (1000 / 2.2) # travel time in ns, using 2200 m/s as speed of sound
# dH2O = 25
# ttH2O = dH2O * (1000 / 1.5)
# tstModel = [(1.5, 1, ttH2O), (2.7, 1, ttPLA), (1.5, 1, ttH2O)]
# # tstCoeffs = tm.calculateReflectionAndTransmissionCoeffs(tstModel, 2)
# # print(tm.propagateWaveThroughLayer(tstWave, tstLayer, 0.5, -0.2))
# # print(tm.propagateWavesThroughLayers(tstWave, tstModel, tstCoeffs, -1, 0.01, 10))
# signals = tm.generateSignalFromModel(ref, refX, tstModel, 'both', 'both',
#                                      3, 0.01, 50000, -1, True)
# # print(signals['echo_forward_sim'])
# # plt.plot(signals['echo_forward_time'], signals['echo_forward'])
# print(signals.keys())
# # plt.show()


################################################
## Turning model into optimization algorithm####
################################################

# calculate leading fit w/ front weighting
#       Need to improve this algorithm to have a better noise tolerance for later finds
#       currently can find first shift where res goes below a threshold - can that threshold be intelligently picked?
#           have a 'minimum separation' parameter - min x-dist allowed between two peaks
#               once res is below threshold, find minimum res w/in min separation
#                 maybe look into convolution up to first maximum of the ref? look for an inflection or extremum?
#           how do we make it noise tolerant - filter residual?
#       Solution: calculate 'leading edge' of ref - first half cycle length
#           calculate convolution w/ signal
#           find tof by threshold
#           find maximum or inflection w/in leading edge len of tof in abs(conv)
#               first look for extrema. if extrema are at edges, look for inflection
#               what if there are multiple maxima? use first?
#               are you sure about inflections? they would catch a plateau in the signal, but is that likely?
#                   at the same time, inflections should be just as fast since 2nd derivative is basically free if using savgol method
#           optimize ref @ extremum
#           may be able to rigorously prove some nice properties by assuming periodicity and symmetry of ref and sig
#               this may also be faster in fourier domain
#           waaaiiittt - I can just do this  with correlations of the leading edge (have I already tried this...?)
#           also this still won't work if the decomposition shifts are too close together

#       alternative:
#           determine minimum shift difference
#           set ref leading edge = ref[0 : min difference]
#           correlation fit leading edge, find first minimum?
#               would fitting derivatives simultaneously help?
#               maybe residual is not the metric - we want to increase penalty of individual point mismatch
#                   what about error to fourth power?
#                   derivative may carry less of this issue since the leading edge is decreasing not increasing
#                       the scenario that is hard to avoid in leading edge is that the point-by-point fit of the neighboring
#                       shift may be not as good, but since the overall signal value is also increasing that may override the worse fit
#                           does this worry actually make sense if residual is not scaled by the signal value? its an absolute, not relative measure
#                       what about normalizing each residual by the values in the fitting window? need to be careful to avoid
#                           adding a different bias in this way -
#                             maybe res/amplitude?
#           fit ref