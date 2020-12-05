import numpy as np
import os
import pickle
import linecache
import tensorflow as tf
import keras
from keras import layers
from keras import backend as K
from keras.layers import *
from keras.backend.tensorflow_backend import set_session
from keras.models import Model, load_model
from keras.layers import Concatenate
from keras import regularizers
import keras.layers.core as core
from keras.layers import Dense,Activation,Convolution2D, Convolution1D, MaxPool2D, Flatten, BatchNormalization, Dropout, Input, Bidirectional, MaxPool1D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from keras.callbacks import ModelCheckpoint
import math
import lightgbm as lgb
import joblib
import argparse


###The ensemble predictor###
class ATPensemble(object):
    def __init__(self,fastapath, pdbid, psiblastoutpath, PSSMpath, psipredoutpath):
        '''
        fastapath: where you put your fasta file
        pdbid: protein id of fasta file
        psiblastoutpath: path for output file of PSI-Blast
        PSSMpath: path for the PSSM file
        psipredoutpath: path for output file of psipred'
        '''        
        self.fastapath = fastapath
        self.pdbid = pdbid
        self.psiblastoutpath = psiblastoutpath
        self.PSSMpath = PSSMpath
        self.psipredoutpath = psipredoutpath
        
    def PSSMfeature(self):
        cmd='/home/songjiazhi/blast/bin/psiblast -evalue 10 -num_iterations 3 -db /home/songjiazhi/blast/db/uniprot -query '+self.fastapath+'/'+self.pdbid+'.fasta'+' -outfmt 0 -out '+self.psiblastoutpath+'/'+self.pdbid+'.fm0 -out_ascii_pssm '+self.PSSMpath+'/'+self.pdbid+'.pssm -num_alignments 1500 -num_threads 8'
        '''
        change to your own PSI-Blast dir
        the referenced database is availabel at $HOME/tool/blast/db/uniprot
        '''
        os.system(cmd)  
        pssmfilelines = linecache.getlines(self.PSSMpath+'/'+self.pdbid+'.pssm')
        pssmDic = {}
        for line in pssmfilelines:
            content = line.split()
            if len(content) == 44:
                residuePosition = int(content[0])-1
                pssmDic[residuePosition] = []
                for i in range(2,22):
                    #pssmDic[str(residuePosition)].append(int(content[i]))
                    pssmDic[residuePosition].append(self.normalize(int(content[i])))     
        return pssmDic
    
    def normalize(self,value):
        a = 1+math.exp(value)
        b = 1/a
        return b        
    
    def psipredfeature(self):
        cmd = '/home/songjiazhi/psipred.4.02/psipred/BLAST+/runpsipredplus '+self.fastapath+'/'+self.pdbid+'.fasta'
        '''change to your psipred dir'''
        os.system(cmd)        
        psipredDic = {}
        filelines = linecache.getlines(self.psipredoutpath+'/'+self.pdbid+'.ss2')
        length = len(filelines)
        for i in range(2,length):
            residuenum = int(filelines[i].split()[0])-1
            psipredDic[residuenum] = []
            psipredDic[residuenum].append(float(filelines[i].split()[3]))
            psipredDic[residuenum].append(float(filelines[i].split()[4]))
            psipredDic[residuenum].append(float(filelines[i].split()[5]))
        return psipredDic   
    
    def onehotfeature(self):
        fastalines = linecache.getlines(self.fastapath+'/'+self.pdbid+'.fasta')
        chemicaldic = {}
        fastaline = fastalines[1]
        if fastaline[-1] == '\n':
            fastaline = fastaline[:-1]
        length = len(fastaline)
        for i in range (length):
            residuename = fastaline[i]
            if residuename == 'A' or residuename == 'G' or residuename == 'V':
                chemicaldic[i] = [0,0,0,0,0,0,1]
            elif residuename == 'I' or residuename == 'L' or residuename == 'F' or residuename == 'P':
                chemicaldic[i] = [0,0,0,0,0,1,0]
            elif residuename == 'H' or residuename == 'N' or residuename == 'Q' or residuename == 'W':
                chemicaldic[i] = [0,0,0,0,1,0,0]
            elif residuename == 'Y' or residuename == 'M' or residuename == 'T' or residuename == 'S':
                chemicaldic[i] = [0,0,0,1,0,0,0]
            elif residuename == 'R' or residuename == 'K':
                chemicaldic[i] = [0,0,1,0,0,0,0]
            elif residuename == 'D' or residuename == 'E':
                chemicaldic[i] = [0,1,0,0,0,0,0]
            elif residuename == 'C':
                chemicaldic[i] = [1,0,0,0,0,0,0]
            elif residuename == 'U':
                chemicaldic[i] = [0,0,0,0,0,0,0] 
        return chemicaldic
        
    def featurecombine(self):
        pssmdic = self.PSSMfeature()
        psipreddic = self.psipredfeature()
        chemicaldic = self.onehotfeature()
        length = len(pssmdic.keys())
        featuredic = {}
        for i in range(length):
            featuredic[i] = []
            for each in pssmdic[i]:
                featuredic[i].append(each)
            for each in psipreddic[i]:
                featuredic[i].append(each)
            for each in chemicaldic[i]:
                featuredic[i].append(each) 
        appendedfeaturedic = self.appendzero(17,featuredic)
        combinefeaturedic = self.combine(length, appendedfeaturedic, 17)
        return combinefeaturedic  
    
    def appendzero(self,windowsize,featureDic):
        seqlength = len(featureDic.keys())
        appendnum = int((windowsize+1)/2)
        for i in range(1,appendnum):
            featureDic[0-i] = []
            featureDic[seqlength-1+i] = []
            for a in range(30):
                featureDic[0-i].append(0)
            for b in range(30):
                featureDic[seqlength-1+i].append(0)
        return featureDic    
    
    def combine(self,sequencelength,featuredic,windowsize):
        neighnum = int((windowsize-1)/2)
        combineDic = {}
        for i in range(0,sequencelength):
            combineDic[i] = []
            for a in range(i - neighnum,i + neighnum + 1):
                #combineDic[i].append(pssmdic[a])
                for each in featuredic[a]:
                    combineDic[i].append(each)
        featurelist = []
        for i in range(0,sequencelength):
            featurelist.append(combineDic[i])
        return featurelist     
    
    def seperatefeature(self, featurelist):
        pssmfeature = []
        psipredfeature = []
        chemicalfeature = []
        for each in featurelist:
            pssmfeature_each = []
            psipredfeature_each = []
            chemicalfeature_each = []
            pssmflag = 0
            psipredflag = 20
            chemicalflag = 23
            for i in range(0,17):
                for a in range(pssmflag, pssmflag+20):
                    pssmfeature_each.append(each[a])
                for b in range(psipredflag, psipredflag+3):
                    psipredfeature_each.append(each[b])
                for c in range(chemicalflag, chemicalflag+7):
                    chemicalfeature_each.append(each[c])
                pssmflag = pssmflag+30
                psipredflag = psipredflag+30
                chemicalflag = chemicalflag+30  
            pssmfeature.append(pssmfeature_each)
            psipredfeature.append(psipredfeature_each)
            chemicalfeature.append(chemicalfeature_each)
        seperatedfeature = (pssmfeature, psipredfeature, chemicalfeature)
        return seperatedfeature 
    
    def multiincepresnetprediction(self, seperatedfeature):
        pssmfeature_test = seperatedfeature[0]
        pssmfeature_test = np.array(pssmfeature_test)
        pssmfeature_test = pssmfeature_test.reshape(-1,17,20,1)
        psipredfeature_test = seperatedfeature[1]
        psipredfeature_test = np.array(psipredfeature_test)
        psipredfeature_test = psipredfeature_test.reshape(-1,17,3,1)
        onehotfeature_test = seperatedfeature[2]
        onehotfeature_test = np.array(onehotfeature_test)
        onehotfeature_test = onehotfeature_test.reshape(-1,17,7,1)
        multiincepresnetmodel = load_model('/home/songjiazhi/atpcapsule/paper/model/multiincepresnetmodel.hdf5')
        '''change to your path where you save the folder. e.g. $HOME/ATPensemble/model/multiincepresnetmodel.hdf5'''
        prediction = multiincepresnetmodel.predict([pssmfeature_test, psipredfeature_test, onehotfeature_test])
        length = len(prediction)
        predict = []
        for i in range(length):
            predict.append(prediction[i][1])
        return predict
    
    def multixceptionprediction(self, seperatedfeature):
        pssmfeature_test = seperatedfeature[0]
        pssmfeature_test = np.array(pssmfeature_test)
        pssmfeature_test = pssmfeature_test.reshape(-1,17,20,1)
        psipredfeature_test = seperatedfeature[1]
        psipredfeature_test = np.array(psipredfeature_test)
        psipredfeature_test = psipredfeature_test.reshape(-1,17,3,1)
        onehotfeature_test = seperatedfeature[2]
        onehotfeature_test = np.array(onehotfeature_test)
        onehotfeature_test = onehotfeature_test.reshape(-1,17,7,1)
        multixceptionmodel = load_model('/home/songjiazhi/atpcapsule/paper/model/multixceptionmodel.hdf5')
        '''change to your path where you save the folder. e.g. $HOME/ATPensemble/model/multixceptionmodel.hdf5'''
        prediction = multixceptionmodel.predict([pssmfeature_test, psipredfeature_test, onehotfeature_test])
        length = len(prediction)
        predict = []
        for i in range(length):
            predict.append(prediction[i][1])
        return predict  
    
    def lgbprediction(self, combinefeaturedic):
        model = joblib.load('/home/songjiazhi/atpcapsule/paper/lgb_model.m')
        lightgbmprediction = model.predict(combinefeaturedic)
        return lightgbmprediction
    
    def predictionensemble(self, multiincepresnetprediction, multixceptionprediction, lgbprediction):
        length = len(multiincepresnetprediction)
        ensemblepredict = []
        for i in range(length):
            ensemblepredict.append(multiincepresnetprediction[i]*0.2+multixceptionprediction[i]*0.2+lgbprediction[i]*0.6)
        return ensemblepredict
    
###The sequence template-based predictor###
class ATPensembletemplate(object):
    def __int__(self, fastapath, pdbid, psiblastoutpath):
        '''
        fastapath: where you put your fasta file
        pdbid: protein id of fasta file
        psiblastoutpath: path for output file of PSI-Blast for sequence template-based predictor
        '''        
        self.fastapath = fastapath
        self.pdbid = pdbid
        self.psiblastoutpath = psiblastoutpath        
        
    def runpsiblast(self):
        cmd = '/home/songjiazhi/blast/bin/psiblast -evalue 10 -num_iterations 3 -db /home/songjiazhi/blast/db/template -query '+self.fastapath+'/'+self.pdbid+'.fasta'+' -outfmt 0 -out '+self.psiblastoutpath+'/'+self.pdbid+'.fm0  -num_alignments 1500 -num_threads 8'
        '''
            change to your own PSI-Blast dir
            the referenced database is availabel at $HOME/atpbinding/templateblastdb/template
            '''
        os.system(cmd)
        
    def parser(self):
        filelines = linecache.getlines(self.psiblastoutpath+'/'+self.pdbid+'.fm0')
        length = len(filelines)
        #print(length)
        i = 0
        seq = {}
        blastseq = {}
        querysequence = ''
        alignsequence = ''
        partnerid = ''
        score = ''
        while i < length:
            if filelines[i] == '' or filelines[i][0] != '>':
                i = i+1
            else:
                break
        partnerid = filelines[i].split()[0][1:]
        scoreline = filelines[i+3]
        score = float(scoreline.split()[2])
        contentnum = i+6
        while filelines[contentnum] != '\n':
            seqid = filelines[contentnum].split()[1]
            seq[int(seqid)] = filelines[contentnum].split()[2]
            blastid = filelines[contentnum+2].split()[1]
            blastseq[int(blastid)] = filelines[contentnum+2].split()[2]
            contentnum = contentnum+4
        seqIdList = []
        blastIdList = []
        seqsegment = sorted(seq.keys())
        blastsegment = sorted(blastseq.keys())
        segmentlen = len(seqsegment)
        segnum = 0
        while segnum < segmentlen:
            seqBegin = seqsegment[segnum]
            blastSeqBegin = blastsegment[segnum]
            SegmentSeq = seq[seqBegin]
            blastSegmentSeq = blastseq[blastSeqBegin]
            seqlength = len(SegmentSeq)
            for i in range(0,seqlength):
                if SegmentSeq[i] == blastSegmentSeq[i]:
                    if SegmentSeq[0] != '-':
                        resSeqId = seqBegin-1+i-SegmentSeq[:i].count('-')
                    else:
                        t = 0
                        while SegmentSeq[t] == '-':
                            t = t+1
                        resSeqId = seqBegin-1+i-t-SegmentSeq[t:i].count('-')
                    seqIdList.append(resSeqId)
                    if blastSegmentSeq[0] != '-':
                        blastSeqId = blastSeqBegin-1+i-blastSegmentSeq[:i].count('-')
                    else:
                        t = 0
                        while blastSegmentSeq[t] == '-':
                            t = t+1
                        blastSeqId = blastSeqBegin-1+i-t-blastsegment[t:i].count('-')
                    blastIdList.append(blastSeqId)
            segnum = segnum+1
        parserdic = {}
        parserdic['partner'] = partnerid
        parserdic['score'] = score
        parserdic['SeqId'] = seqIdList
        parserdic['BlastId'] = blastIdList
        return parserdic   
    
    def siteselection(self,parserdic,sitedicpath):
        sitepickle = open(sitedicpath,'rb')
        sitedic = pickle.load(sitepickle)
        partnerId = parserdic['partner']
        pdbId = partnerId.split('_')[0]
        chainId = partnerId.split('_')[1]
        #site = sitedic[pdbId][chainId]
        site = sitedic[pdbId+'_'+chainId]
        seqIdList = parserdic['SeqId']
        blastIdList = parserdic['BlastId']
        length = len(seqIdList)
        predictedsite = []
        for i in range(0,length):
            #if str(blastIdList[i]) in site.keys():
            if blastIdList[i] in site:
                predictedsite.append(seqIdList[i])
        predict = {}
        predict['score'] = parserdic['score']
        predict['SeqId'] = predictedsite
        return predict

    def templateprediction(self, predict):
        fastalines = linecache.getlines(self.fastapath+'/'+self.pdbid+'.fasta')
        fastaline = fastalines[1]
        if fastaline[-1] == '\n':
            fastaline = fastaline[:-1]
        length = len(fastaline)    
        templatepredictionlist = []
        if predict['score'] < 50:
            for i in range(length):
                templatepredictionlist.append(0)
        else:
            for i in range(length):
                if i in predict['SeqId']:
                    templatepredictionlist.append(1)
                else:
                    templatepredictionlist.append(0)
        return templatepredictionlist    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fasta', help='fasta file path')
    parser.add_argument('-i', '--pdbid', help='fasta file id')
    args = parser.parse_args()
    
    fastapath = args.fasta
    pdbid = args.pdbid
    predict = ATPensemble(fastapath, pdbid, '/home/songjiazhi/atpcapsule/executive/blastout/out', '/home/songjiazhi/atpcapsule/executive/blastout/pssm', '/home/songjiazhi/atpcapsule/executive')
    '''change dirs according to the definations in ATPensemble'''
    featurelist = predict.featurecombine()
    seperatedfeature = predict.seperatefeature(featurelist)  
    multiincepresnetprediction = predict.multiincepresnetprediction(seperatedfeature)
    multixceptionprediction = predict.multixceptionprediction(seperatedfeature)
    lgbprediction = predict.lgbprediction(featurelist)
    prediction = predict.predictionensemble(multiincepresnetprediction, multixceptionprediction, lgbprediction)
    templatepredict = ATPensembletemplate(fastapath, pdbid, '/home/songjiazhi/atpcapsule/executive/templateblastout')
    '''change dirs according to the definations in ATPensembleTemplate'''
    templatepredict.runpsiblast()
    blastoutdic = templatepredict.parser()
    templatesite = templatepredict.siteselection(blastoutdic, '/home/songjiazhi/atpcapsule/sitedic.pickle')
    templateprediction = templatepredict.templateprediction(templatesite)
    length = len(prediction)
    predictfile = open('/home/songjiazhi/atpcapsule/executive/'+pdbid+'_prediction.txt','w')    
    predictfile.write('residue'+'  '+'ensemble'+'  '+'template')
    predictfile.write('\n')
    for i in range(length):
        predictfile.write(str(i)+' '+str(prediction[i])+' '+str(templateprediction[i]))
        predictfile.write('\n')
    predictfile.close()    
    

if __name__ == "__main__":
    main()
    
    

    
    