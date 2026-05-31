from numpy import matlib
from BOA import BOA
from Global_Vars import Global_Vars
from Image_Results import *
from Model_AAHVBRA import Model_AAHVBRA
from Model_DCGFPNRF import Model_DCGFPNRF
from Model_GRU import Model_GRU
from Model_SPDNet import Model_SPDNet
from Model_VGG19 import Model_VGG19
from Objfun import objfun
from POA import POA
from Plot_Results import *
from Proposed import Proposed
from TFMOA import TFMOA
from TSA import TSA

# Read Dataset
an = 0
if an == 1:
    Image = []
    FileName = './Dataset/Sec Growth DataScience staff meeting Sep 14 2022 [rOqgRiNMVqg].f398.mp4'
    cap = cv.VideoCapture(FileName)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        ReImg = cv.resize(frame, (512, 512))
        Image.append(ReImg)
    np.save('Image.npy', np.asarray(Image))

# Face Detection
an = 0
if an == 1:
    Image = np.load('Image.npy', allow_pickle=True)
    Eval, Pred, DetectImg, Target = Model_DCGFPNRF(Image)
    np.save('Eval_all.npy', Eval)
    np.save('Target.npy', Target)
    np.save('DetectedImage.npy', DetectImg)

# Optimization for Classification
an = 0
if an == 1:
    Image = np.load('DetectedImage.npy', allow_pickle=True)  # Loading Step
    Target = np.load('Target.npy', allow_pickle=True)  # Loading Step
    Global_Vars.Image = Image  # Store loaded data in global variables for access inside objective function
    Global_Vars.Target = Target  # Store loaded data in global variables for access inside objective function
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Learning Rate, No of Epochs
    xmin = matlib.repmat([5, 0.01, 5], Npop, 1)
    xmax = matlib.repmat([255, 0.99, 50], Npop, 1)
    fname = objfun
    initsol = np.zeros((Npop, Chlen))  # Initialize population matrix (solutions)
    # Randomly initialize each individual within bounds
    for p1 in range(initsol.shape[0]):  # Loop over population
        for p2 in range(initsol.shape[1]):  # Loop over parameters
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("TFMOA...")
    [bestfit1, fitness1, bestsol1, time1] = TFMOA(initsol, fname, xmin, xmax, Max_iter)

    print("TSA...")
    [bestfit2, fitness2, bestsol2, time2] = TSA(initsol, fname, xmin, xmax, Max_iter)

    print("BOA...")
    [bestfit3, fitness3, bestsol3, time3] = BOA(initsol, fname, xmin, xmax, Max_iter)

    print("POA...")
    [bestfit4, fitness4, bestsol4, time4] = POA(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BestSol.npy', BestSol)  # Save BestSol

# Classification
an = 0
if an == 1:
    Feat = np.load('DetectedImage.npy', allow_pickle=True)  # loading step
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    BestSol = np.load('BestSol.npy', allow_pickle=True)  # loading step
    EVAL = []
    LearnPer = [35, 45, 55, 65]  # For Comparison
    for act in range(len(LearnPer)):
        learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_AAHVBRA(Train_Data, Train_Target, Test_Data,
                                             Test_Target, sol)  # With optimization
        Eval[5, :], pred1 = Model_GRU(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred2 = Model_SPDNet(Train_Data, Train_Target, Test_Data,
                                         Test_Target)
        Eval[7, :], pred3 = Model_VGG19(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred4 = Model_AAHVBRA(Train_Data, Train_Target, Test_Data,
                                          Test_Target)  # Without optimization
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Evaluate_all.npy', EVAL)  # Save Eval all

plotConvResults()
Plot_Results()
Plot_ROC_Curve()
Table()
Image_Results()
Sample_Images()
