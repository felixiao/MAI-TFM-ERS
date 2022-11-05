import pandas as pd
import os
import glob

class ProcessResult():
    def __init__(self,path,model='PETER+'):
        """_summary_

        Args:
            path (str): the folder to the log files
            name (str): the template log file name, eg. TA_usefeat_{}.log where {} will be replaced with the subsets (1-5)
        """        
        self.path = path
        self.model = model
    
    def LoadOne(self,filename):
        result = {'sub':int(filename[-5:-4]),'train':{'loss':[],'val_loss':[],'textppl':[],'val_textppl':[],'rating_loss':[],'val_rating_loss':[]},'test':{}}
        with open(filename,'r') as f:
            lns = f.readlines()
            for l in lns:
                if self.model in ['PETER','PETER+']:
                    if 'valid' in l:
                        loss = l.split('|')[2][13:-1]
                        val_loss = l.split('|')[-1][12:-15]
                        # print('loss=',float(loss),'val_loss=',float(val_loss))
                        result['train']['loss'].append(float(loss))
                        result['train']['val_loss'].append(float(val_loss))
                    if 'test' in l:
                        result['test']['context ppl'] = float(l.split('|')[0].split(' ')[-2])
                        result['test']['text ppl'] = float(l.split('|')[1].split(' ')[-2])
                        result['test']['rating loss'] = float(l.split('|')[2].split(' ')[3])
                        # print('context ppl=',result['test']['context ppl'],'text ppl=',result['test']['text ppl'],'rating loss=',result['test']['rating loss'])
                elif self.model == 'NRT':
                    if 'on train' in l:
                        loss = l.split('|')[-1].split(' ')[3]
                        # print('loss=',float(loss))
                        result['train']['loss'].append(float(loss))
                        txtppl = l.split('|')[0].split(' ')[-2]
                        # print('txtppl=',txtppl)
                        result['train']['textppl'].append(float(txtppl))
                        rating_loss = l.split('|')[1].split(' ')[-2]
                        # print('rating loss=',rating_loss)
                        result['train']['rating_loss'].append(float(rating_loss))
                    if 'on validation' in l:
                        val_loss = l.split('|')[-1].split(' ')[3]
                        # print('val_loss=',float(val_loss))
                        result['train']['val_loss'].append(float(val_loss))
                        txtppl = l.split('|')[0].split(' ')[-2]
                        # print('val_txtppl=',txtppl)
                        result['train']['val_textppl'].append(float(txtppl))
                        rating_loss = l.split('|')[1].split(' ')[-2]
                        # print('val_rating loss=',rating_loss)
                        result['train']['val_rating_loss'].append(float(rating_loss))
                    if 'test' in l:
                        result['test']['text ppl'] = float(l.split('|')[0].split(' ')[-2])
                        result['test']['rating loss'] = float(l.split('|')[1].split(' ')[-2])
                        result['test']['total loss'] = float(l.split('|')[2].split(' ')[3])
                elif self.model == 'Att2Seq':
                    if 'on train' in l:
                        loss = l.split(' ')[-3]
                        # print('loss=',float(loss))
                        result['train']['textppl'].append(float(loss))
                    if 'on validation' in l:
                        val_loss = l.split(' ')[-3]
                        # print('val_loss=',float(val_loss))
                        result['train']['val_textppl'].append(float(val_loss))
                    if 'End of' in l:
                        result['test']['text ppl'] = float(l.split('|')[0].split(' ')[-2])
                        # print('text ppl=',result['test']['text ppl'],'total loss=',result['test']['total loss'],'rating loss=',result['test']['rating loss'])
                if 'RMSE' in l:
                    result['test']['RMSE↓'] = float(l.split('RMSE ')[-1][:-1])
                    # print('RMSE=',result['test']['RMSE'])
                if 'MAE' in l:
                    result['test']['MAE↓'] = float(l.split('MAE ')[-1][:-1])
                    # print('MAE=',result['test']['MAE'])
                if 'BLEU-1' in l:
                    result['test']['B1↑'] = float(l.split('BLEU-1 ')[-1][:-1])
                    # print('BLEU-1=',result['test']['B1'])
                if 'BLEU-4' in l:
                    result['test']['B4↑'] = float(l.split('BLEU-4 ')[-1][:-1])
                    # print('BLEU-4=',result['test']['B4'])
                if 'USR' in l:
                    result['test']['USR↑'] = float(l.split('|')[0].split('USR')[-1][:-1])
                    result['test']['USN↑'] = float(l.split('|')[1].split('USN')[-1][:-1])
                    # print('USR=',result['test']['USR'],'USN=',result['test']['USN'])
                if 'DIV' in l:
                    result['test']['DIV↓'] = float(l.split('DIV ')[-1][:-1])
                    # print('DIV=',result['test']['DIV'])
                if 'FCR' in l:
                    result['test']['FCR↑'] = float(l.split('FCR ')[-1][:-1])
                    # print('FCR=',result['test']['FCR'])
                if 'FMR' in l:
                    result['test']['FMR↑'] = float(l.split('FMR ')[-1][:-1])
                    # print('FMR=',result['test']['FMR'])
                if 'rouge_1/f_score' in l:
                    result['test']['R1-F↑'] = float(l.split('rouge_1/f_score ')[-1][:-1])
                    # print('rouge_1/f_score=',result['test']['R1-F'])
                if 'rouge_1/r_score' in l:
                    result['test']['R1-R↑'] = float(l.split('rouge_1/r_score ')[-1][:-1])
                    # print('rouge_1/r_score=',result['test']['R1-R'])
                if 'rouge_1/p_score' in l:
                    result['test']['R1-P↑'] = float(l.split('rouge_1/p_score ')[-1][:-1])
                    # print('rouge_1/p_score=',result['test']['R1-P'])

                if 'rouge_2/f_score' in l:
                    result['test']['R2-F↑'] = float(l.split('rouge_2/f_score ')[-1][:-1])
                    # print('rouge_2/f_score=',result['test']['R2-F'])
                if 'rouge_2/r_score' in l:
                    result['test']['R2-R↑'] = float(l.split('rouge_2/r_score ')[-1][:-1])
                    # print('rouge_2/r_score=',result['test']['R2-R'])
                if 'rouge_2/p_score' in l:
                    result['test']['R2-P↑'] = float(l.split('rouge_2/p_score ')[-1][:-1])
                    # print('rouge_2/p_score=',result['test']['R2-P'])

                if 'rouge_l/f_score' in l:
                    result['test']['RL-F↑'] = float(l.split('rouge_l/f_score ')[-1][:-1])
                    # print('rouge_l/f_score=',result['test']['RL-F'])
                if 'rouge_l/r_score' in l:
                    result['test']['RL-R↑'] = float(l.split('rouge_l/r_score ')[-1][:-1])
                    # print('rouge_l/r_score=',result['test']['RL-R'])
                if 'rouge_l/p_score' in l:
                    result['test']['RL-P↑'] = float(l.split('rouge_l/p_score ')[-1][:-1])
                    # print('rouge_l/p_score=',result['test']['RL-P'])
                
            f.close()
        # print(result)
        return result
    
    def LoadAll(self):
        nfiles = glob.glob(self.path+'*.log')
        # print(nfiles)
        res = []
        ind = []
        for f in nfiles:
            d = self.LoadOne(f)
            ind.append(d['sub'])
            res.append(d['test'])
        dt = pd.DataFrame(data=res,index=ind)
        dtmean = dt.mean()
        return dt,dtmean


def CompareResults(path_csv, parent_path,models,paper=True):
    CmpRes = []
    for i,p in enumerate(models):
        pr = ProcessResult(os.path.join(parent_path,p)+'/',p)
        _,pr_mean = pr.LoadAll()
        # print(pr_mean)
        CmpRes.append(pr_mean)
    name_len = len(parent_path.split(os.sep)[-1])
    print('\n'+'='*20+' '*(12-name_len//2) + parent_path.split(os.sep)[-1] +' '*(12-name_len//2)+'='*20)
    table = pd.DataFrame(data=CmpRes,index=models)
    if paper:
        paper_df = pd.read_csv(os.path.join(parent_path,'Paper_'+path_csv),index_col=0)
        # print(paper_df)
        mp = [m+'_P' for m in models]
        mo = []
        for i in range(len(models)) :
            mo.append(models[i])
            mo.append(mp[i])
        table = pd.concat([table,paper_df.loc[mp]])
        print(table[['RMSE↓','MAE↓','FMR↑','FCR↑','DIV↓','USR↑','B1↑','B4↑','R1-P↑','R1-R↑','R1-F↑','R2-P↑','R2-R↑','R2-F↑']].loc[mo])
    else:
        print(table[['RMSE↓','MAE↓','FMR↑','FCR↑','DIV↓','USR↑','B1↑','B4↑','R1-P↑','R1-R↑','R1-F↑','R2-P↑','R2-R↑','R2-F↑']])
    table.to_csv(os.path.join(parent_path,path_csv))

CompareResults('TA_RESULT_TABLE.csv','./Result/TripAdvisor',['NRT','Att2Seq','PETER','PETER+'])
CompareResults('CSJ_RESULT_TABLE.csv','./Result/Amazon/ClothingShoesAndJewelry',['NRT','Att2Seq','PETER','PETER+'],False)
CompareResults('MT_RESULT_TABLE.csv','./Result/Amazon/MoviesAndTV',['NRT','Att2Seq','PETER+'])
CompareResults('Yelp_RESULT_TABLE.csv','./Result/Yelp',['PETER+'])
