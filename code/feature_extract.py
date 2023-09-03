# coding=utf-8
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from global_v import *

def processing_feature(file,dataset_num):
    metric= pd.DataFrame()
    feats = {"id" : file}
    if os.path.exists(f"./data/{dataset_num}/metric/{file}_metric.csv"):
        metric = pd.read_csv(f"./data/{dataset_num}/metric/{file}_metric.csv").reset_index(drop=True)            
    if len(metric)>0:
        feats['m_l']=len(metric)
        metric.drop_duplicates(inplace=True)
        feats['m_l_nodup'] = len(metric) 
        metric['tags_dict']=metric['tags'].apply(lambda x:eval(x))
        metric['tags_dict_len']=metric['tags_dict'].apply(lambda x: len(x))
        for n in metric_tags_keys:
            metric[f'tag_{n}']=metric['tags_dict'].apply(lambda x: x[f'{n}'] if n in x.keys() else None)
            feats[f'm_tag_{n}_n']=metric[f'tag_{n}'].nunique()
            feats[f'm_tag_{n}_c']=metric[f'tag_{n}'].count()
        metric_exit_tag_name=set(metric['tag_metric_name'])
        metric_exit_service_name=set(metric['tag_service_name'])
        metric_exit_timestamp=set(metric['timestamp'])
        
        # metric_service_4           ['cpm','resp_time','error_count','success_rate']
        metric_4=metric[metric['tag_metric_name'].isin(metric_tags_4)]
        for n in metric_tags_4:
            if n in metric_exit_tag_name:
                for sn in metric_tag_service_name_test:
                    if sn in metric_exit_service_name:
                        temp=metric_4[(metric_4['tag_metric_name']==n)&(metric_4['tag_service_name']==sn)]
                        feats[f'm_tag_{n}_service{sn}_ts_len']=len(temp)
                        for stats_func in ['mean','min','max','ptp','median']:
                            feats[f'm_tag_{n}_service{sn}_v_{stats_func}']=temp['value'].agg(stats_func) 
                        feats[f'm_tag_{n}_service{sn}_v_meanmin']=feats[f'm_tag_{n}_service{sn}_v_mean']-feats[f'm_tag_{n}_service{sn}_v_min']
                        feats[f'm_tag_{n}_service{sn}_v_maxmean']=feats[f'm_tag_{n}_service{sn}_v_max']-feats[f'm_tag_{n}_service{sn}_v_mean']
                    else:
                        feats[f'm_tag_{n}_service{sn}_ts_len']=-1
                        for stats_func in ['mean','min','max','ptp','median']:
                            feats[f'm_tag_{n}_service{sn}_v_{stats_func}']=-1
                        feats[f'm_tag_{n}_service{sn}_v_meanmin']=-1
                        feats[f'm_tag_{n}_service{sn}_v_maxmean']=-1
            else:
                for sn in metric_tag_service_name_test:
                    feats[f'm_tag_{n}_service{sn}_ts_len']=-2
                    for stats_func in ['mean','min','max','ptp','median']:
                        feats[f'm_tag_{n}_service{sn}_v_{stats_func}']=-2    
                    feats[f'm_tag_{n}_service{sn}_v_meanmin']=-2
                    feats[f'm_tag_{n}_service{sn}_v_maxmean']=-2                                 
        
        # metric_container15
        metric_container=metric[metric['tag_metric_name'].isin(metric_tags_15)]
        metric_container_exit_container=metric_container[f'tag_container'].unique()
        metric_container_exit_instance=metric_container[f'tag_instance'].unique()
        metric_container[f'tag_instance_num']=metric_container[f'tag_instance'].apply(lambda x: metric_container_instance_str2num(x))    
        metric_container[f'tag_pod_n']=metric_container[f'tag_pod'].apply(lambda x: x.split('-')[0]+'-'+x.split('-')[1] if len(x.split('-'))>2  else x)
        feats['m_tag_container_job_n']=metric_container['tag_job'].nunique()
        for ins in metric_tag_instance_nsdefault:
            if ins in  metric_container_exit_instance:
                feats[f'm_tag_instance{ins}_len']=len(metric_container[metric_container[f'tag_instance']==ins])   
                feats[f'm_tag_instance{ins}_pod_n']=metric_container[metric_container[f'tag_instance']==ins]['tag_pod_n'].nunique()    
                feats[f'm_tag_instance{ins}_container_n']=metric_container[metric_container[f'tag_instance']==ins]['tag_container'].nunique()    
            else:
                feats[f'm_tag_instance{ins}_len']=-1   
                feats[f'm_tag_instance{ins}_pod_n']=-1      
                feats[f'm_tag_instance{ins}_container_n']=-1                   
        for ct in metric_container_80:
            if ct in metric_container_exit_container:
                feats[f'm_tag_contain_contain{ct}_len']=len(metric_container[metric_container[f'tag_container']==ct])           
            else:
                feats[f'm_tag_contain_contain{ct}_len']=-1                                                   
        for sn in metric_pod80:
            if sn in  metric_container['tag_pod_n'].unique():
                feats[f'm_pod{sn}_nodeinstance']=metric_container[metric_container['tag_pod_n']==sn]['tag_instance_num'].unique().mean() 
                feats[f'm_pod{sn}_ts_len']=len(metric_container[metric_container['tag_pod_n']==sn])
            else:
                feats[f'm_pod{sn}_nodeinstance']=-1
                feats[f'm_pod{sn}_ts_len']=-1      
        for n in metric_tag_contain_network8:
            temp_n=metric_container[(metric_container['tag_metric_name']==n)]
            temp_n_exit_pod=temp_n['tag_pod_n'].unique()
            if n in metric_exit_tag_name:
                for sn in metric_tag_net8_4instance_pod7:
                    if sn in  temp_n_exit_pod: 
                        temp=temp_n[metric_container['tag_pod_n']==sn]
                        for interface in metric_contain_net8_pod2interface[f'{sn}']:
                            if interface in temp['tag_interface'].unique(   ):
                                for ins in metric_tag_instance_nsdefault:
                                    for stats_func in ['mean','min','max','ptp']:
                                        feats[f'm_tag_{n}_pod{sn}_interface{interface}_ins{ins}_v_{stats_func}']=temp[(temp['tag_interface']==interface)&(temp['tag_instance']==ins)]['value'].agg(stats_func) 
                            else:
                                for ins in metric_tag_instance_nsdefault:
                                    for stats_func in ['mean','min','max','ptp']:
                                        feats[f'm_tag_{n}_pod{sn}_interface{interface}_ins{ins}_v_{stats_func}']=-1
                    else:
                        for interface in metric_contain_net8_pod2interface[f'{sn}']:
                            for ins in metric_tag_instance_nsdefault:
                                for stats_func in ['mean','min','max','ptp']:
                                    feats[f'm_tag_{n}_pod{sn}_interface{interface}_ins{ins}_v_{stats_func}']=-2   
                for sn in metric_pod80:
                    if sn in  temp_n_exit_pod: 
                        temp=temp_n[metric_container['tag_pod_n']==sn]
                        for interface in metric_contain_net8_pod2interface[f'{sn}']:
                            if interface in temp['tag_interface'].unique(   ):
                                for stats_func in ['mean','min','max','ptp']:
                                    feats[f'm_tag_{n}_pod{sn}_interface{interface}_v_{stats_func}']=temp[temp['tag_interface']==interface]['value'].agg(stats_func) 
                            else:
                                for stats_func in ['mean','min','max','ptp']:
                                    feats[f'm_tag_{n}_pod{sn}_interface{interface}_v_{stats_func}']=-1
                    else:
                        for interface in metric_contain_net8_pod2interface[f'{sn}']:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_pod{sn}_interface{interface}_v_{stats_func}']=-2                        
            else:
                for sn in metric_tag_net8_4instance_pod7:
                    for interface in metric_contain_net8_pod2interface[f'{sn}']:
                        for ins in metric_tag_instance_nsdefault:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_pod{sn}_interface{interface}_ins{ins}_v_{stats_func}']=-3   
                for sn in metric_pod80:
                    for interface in metric_contain_net8_pod2interface[f'{sn}']:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_pod{sn}_interface{interface}_v_{stats_func}']=-3
        for n in metric_tag_cpu7_2container_3:
            ##  container ‘’ 
            temp_n=metric_container[(metric_container['tag_metric_name']==n)&(metric_container['tag_container']=='')]
            if len(temp_n)>0:
                for pod in metric_tag_cpu7_2container_3_empty_4instance1_pod1:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_pod{pod}_ts_0len']=len(temp)
                        for node in metric_tag_instance_nsdefault:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_pod{pod}_node{node}_v_{stats_func}']=temp[temp['tag_instance']==node]['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_pod{pod}_ts_0len']=-1
                        for node in metric_tag_instance_nsdefault:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_pod{pod}_node{node}_v_{stats_func}']=-1 
                for pod in metric_tag_cpu7_2container_3_emptyct_pod53:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_pod{pod}_ts_len']=len(temp)
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_pod{pod}_v_{stats_func}']=temp['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_pod{pod}_ts_len']=-1
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_pod{pod}_v_{stats_func}']=-1 
            else:
                for pod in metric_tag_cpu7_2container_3_empty_4instance1_pod1:
                    feats[f'm_tag_{n}_pod{pod}_ts_0len']=-2
                    for node in metric_tag_instance_nsdefault:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_pod{pod}_node{node}_v_{stats_func}']=-2 
                for pod in metric_tag_cpu7_2container_3_emptyct_pod53:
                    feats[f'm_tag_{n}_pod{pod}_ts_len']=-2
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_pod{pod}_v_{stats_func}']=-2
            ##  container not empty
            temp_n=metric_container[(metric_container['tag_metric_name']==n)&(metric_container['tag_container']!='')]
            if len(temp_n)>0:
                ###   pod='nacosdb-mysql'
                pod='nacosdb-mysql'
                if  len(temp_n[temp_n['tag_pod_n']=='nacosdb-mysql'])>0:             
                    temp=temp_n[temp_n['tag_pod_n']==pod]
                    feats[f'm_tag_{n}_notpod{pod}_ts_0len']=len(temp)
                    for node in metric_tag_instance_nsdefault:
                        for ct in ['mysql','xenon']:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_node{node}_ct{ct}_v_{stats_func}']=temp[(temp['tag_instance']==node)&(temp['tag_container']==ct)]['value'].agg(stats_func)       
                else :
                    feats[f'm_tag_{n}_notpod{pod}_ts_0len']=-1
                    for node in metric_tag_instance_nsdefault:
                        for ct in ['mysql','xenon']:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_node{node}_ct{ct}_v_{stats_func}']=-1 
                ### pod=='kube-flannel'
                pod='kube-flannel'
                if  len(temp_n[temp_n['tag_pod_n']=='kube-flannel'])>0:             
                    temp=temp_n[temp_n['tag_pod_n']==pod]
                    feats[f'm_tag_{n}_notpod{pod}_ts_1len']=len(temp)
                    for node in metric_tag_instance_nsdefault:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_node{node}_v_{stats_func}']=temp[(temp['tag_instance']==node)]['value'].agg(stats_func)       
                else :
                    feats[f'm_tag_{n}_notpod{pod}_ts_1len']=-1
                    for node in metric_tag_instance_nsdefault:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_node{node}_v_{stats_func}']=-1           
                ### pod in ['7f1cbe89dd024b6ebbf5556426b34acf', 'c019762f3cbd493cb4dc4443eec2c273', 'ab1c1a0046754e49b4fdf64708124365']
                for pod in metric_tag_cpu7_2container_3_notempty2ct_1instance_pod3:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_notpod{pod}_ts_2len']=len(temp)
                        for ct in ['mysql','xenon']:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_ct{ct}_v_{stats_func}']=temp[(temp['tag_container']==ct)]['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_notpod{pod}_ts_2len']=-1
                        for ct in ['mysql','xenon']:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_ct{ct}_v_{stats_func}']=-1 
                ### other pod
                for pod in metric_tag_cpu7_2container_3_notemptyct_pod57:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_notpod{pod}_ts_len']=len(temp)
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_v_{stats_func}']=temp['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_notpod{pod}_ts_len']=-1
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_v_{stats_func}']=-1 
            else:
                ###   pod='nacosdb-mysql'
                pod='nacosdb-mysql'
                feats[f'm_tag_{n}_notpod{pod}_ts_0len']=-2
                for node in metric_tag_instance_nsdefault:
                    for ct in ['mysql','xenon']:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_node{node}_ct{ct}_v_{stats_func}']=-2
                ### pod=='kube-flannel'
                pod='kube-flannel'
                feats[f'm_tag_{n}_notpod{pod}_ts_1len']=-2
                for node in metric_tag_instance_nsdefault:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_notpod{pod}_node{node}_v_{stats_func}']=-2
                ### pod in ['7f1cbe89dd024b6ebbf5556426b34acf', 'c019762f3cbd493cb4dc4443eec2c273', 'ab1c1a0046754e49b4fdf64708124365']
                for pod in metric_tag_cpu7_2container_3_notempty2ct_1instance_pod3:
                    feats[f'm_tag_{n}_notpod{pod}_ts_2len']=-2
                    for ct in ['mysql','xenon']:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_ct{ct}_v_{stats_func}']=-2
                ### other pod
                for pod in metric_tag_cpu7_2container_3_notemptyct_pod57:
                    feats[f'm_tag_{n}_notpod{pod}_ts_len']=-2
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_notpod{pod}_v_{stats_func}']=-2
        for n in metric_tag_cpu7_3container_4:
            ##  container ‘’ 
            temp_n=metric_container[(metric_container['tag_metric_name']==n)&(metric_container['tag_container']=='')]
            if len(temp_n)>0:
                for pod in metric_tag_cpu7_3container_4_4_instance7:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_pod{pod}_ts_len']=len(temp)
                        for node in metric_tag_instance_nsdefault:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_pod{pod}_node{node}_v_{stats_func}']=temp[temp['tag_instance']==node]['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_pod{pod}_ts_len']=-1
                        for node in metric_tag_instance_nsdefault:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_pod{pod}_node{node}_v_{stats_func}']=-1 
                for pod in metric_contain_cpu7_instance73:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_pod{pod}_ts_len']=len(temp)
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_pod{pod}_v_{stats_func}']=temp['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_pod{pod}_ts_len']=-1
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_pod{pod}_v_{stats_func}']=-1 
            else:
                for pod in metric_tag_cpu7_3container_4_4_instance7:
                    feats[f'm_tag_{n}_pod{pod}_ts_len']=-2
                    for node in metric_tag_instance_nsdefault:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_pod{pod}_node{node}_v_{stats_func}']=-2 
                for pod in metric_contain_cpu7_instance73:
                    feats[f'm_tag_{n}_pod{pod}_ts_len']=-2
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_pod{pod}_v_{stats_func}']=-2
            ##  container POD
            temp_n=metric_container[(metric_container['tag_metric_name']==n)&(metric_container['tag_container']=='POD')]
            if len(temp_n)>0:
                for pod in metric_tag_cpu7_3container_4_4_instance7:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_podpod{pod}_ts_len']=len(temp)
                        for node in metric_tag_instance_nsdefault:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_podpod{pod}_node{node}_v_{stats_func}']=temp[temp['tag_instance']==node]['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_podpod{pod}_ts_len']=-1
                        for node in metric_tag_instance_nsdefault:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_podpod{pod}_node{node}_v_{stats_func}']=-1 
                for pod in metric_contain_cpu7_instance73:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_podpod{pod}_ts_len']=len(temp)
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_podpod{pod}_v_{stats_func}']=temp['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_podpod{pod}_ts_len']=-1
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_podpod{pod}_v_{stats_func}']=-1 
            else:
                for pod in metric_tag_cpu7_3container_4_4_instance7:
                    feats[f'm_tag_{n}_podpod{pod}_ts_len']=-2
                    for node in metric_tag_instance_nsdefault:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_podpod{pod}_node{node}_v_{stats_func}']=-2 
                for pod in metric_contain_cpu7_instance73:
                    feats[f'm_tag_{n}_podpod{pod}_ts_len']=-2
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_podpod{pod}_v_{stats_func}']=-2
            ##  container other 
            temp_n=metric_container[(metric_container['tag_metric_name']==n)&(metric_container['tag_container']!='POD')&(metric_container['tag_container']!='')]
            if len(temp_n)>0:
                ###   pod='nacosdb-mysql'
                pod='nacosdb-mysql'
                if  len(temp_n[temp_n['tag_pod_n']=='nacosdb-mysql'])>0:             
                    temp=temp_n[temp_n['tag_pod_n']==pod]
                    feats[f'm_tag_{n}_notpod{pod}_ts_0len']=len(temp)
                    for node in metric_tag_instance_nsdefault:
                        for ct in ['mysql','xenon','slowlog']:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_node{node}_ct{ct}_v_{stats_func}']=temp[(temp['tag_instance']==node)&(temp['tag_container']==ct)]['value'].agg(stats_func)       
                else :
                    feats[f'm_tag_{n}_notpod{pod}_ts_0len']=-1
                    for node in metric_tag_instance_nsdefault:
                        for ct in ['mysql','xenon','slowlog']:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_node{node}_ct{ct}_v_{stats_func}']=-1 
                ### pod in ['7f1cbe89dd024b6ebbf5556426b34acf', 'c019762f3cbd493cb4dc4443eec2c273', 'ab1c1a0046754e49b4fdf64708124365']
                for pod in metric_tag_cpu7_2container_3_notempty2ct_1instance_pod3:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_notpod{pod}_ts_1len']=len(temp)
                        for ct in ['mysql','xenon','slowlog']:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_ct{ct}_v_{stats_func}']=temp[(temp['tag_container']==ct)]['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_notpod{pod}_ts_1len']=-1
                        for ct in ['mysql','xenon','slowlog']:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_ct{ct}_v_{stats_func}']=-1 
                ### pod in metric_tag_cpu7_3container_4_4_instance7
                for pod in metric_tag_cpu7_3container_4_4_instance7:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_notpod{pod}_ts_len']=len(temp)
                        for node in metric_tag_instance_nsdefault:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_node{node}_v_{stats_func}']=temp[temp['tag_instance']==node]['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_notpod{pod}_ts_len']=-1
                        for node in metric_tag_instance_nsdefault:
                            for stats_func in ['mean','min','max','ptp']:
                                feats[f'm_tag_{n}_notpod{pod}_node{node}_v_{stats_func}']=-1 
                ### other pod
                for pod in metric_contain_cpu7_instance73:
                    if pod in (temp_n['tag_pod_n'].unique()) :
                        temp=temp_n[temp_n['tag_pod_n']==pod]
                        feats[f'm_tag_{n}_notpod{pod}_ts_len']=len(temp)
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_v_{stats_func}']=temp['value'].agg(stats_func) 
                    else :
                        feats[f'm_tag_{n}_notpod{pod}_ts_len']=-1
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_v_{stats_func}']=-1 
            else:
                ###  pod='nacosdb-mysql'
                pod='nacosdb-mysql'
                feats[f'm_tag_{n}_notpod{pod}_ts_0len']=-2
                for node in metric_tag_instance_nsdefault:
                    for ct in ['mysql','xenon','slowlog']:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_node{node}_ct{ct}_v_{stats_func}']=-2 
                ### pod in ['7f1cbe89dd024b6ebbf5556426b34acf', 'c019762f3cbd493cb4dc4443eec2c273', 'ab1c1a0046754e49b4fdf64708124365']
                for pod in metric_tag_cpu7_2container_3_notempty2ct_1instance_pod3:
                    feats[f'm_tag_{n}_notpod{pod}_ts_1len']=-2
                    for ct in ['mysql','xenon','slowlog']:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_ct{ct}_v_{stats_func}']=-2 
                ### pod in metric_tag_cpu7_3container_4_4_instance7
                for pod in metric_tag_cpu7_3container_4_4_instance7:
                    feats[f'm_tag_{n}_notpod{pod}_ts_len']=-2
                    for node in metric_tag_instance_nsdefault:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_notpod{pod}_node{node}_v_{stats_func}']=-2 
                ### other pod
                for pod in metric_contain_cpu7_instance73:
                    feats[f'm_tag_{n}_notpod{pod}_ts_len']=-2
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_notpod{pod}_v_{stats_func}']=-2 

        # metric_node94
        metric_node=metric[metric['tag_metric_name'].isin(metric_tag_node_94)]
        metric_node_exit_instance=metric_node['tag_instance'].unique()
        feats[f'm_tag_node_instance_n']=metric_node['tag_instance'].nunique()
        feats[f'm_tag_node_job_n']=metric_node['tag_job'].nunique()
        feats[f'm_tag_node_kubernetes_pod_name_n']=metric_node['tag_kubernetes_pod_name'].nunique()
        for job in metric_node_jobs3:
            if job in metric_node['tag_job'].unique():
                feats[f'm_tag_node_job{job}']=1
            else:
                feats[f'm_tag_node_job{job}']=0
        for instan in metric_tag_node_instance44:
            if instan in metric_node_exit_instance:
                feats[f'm_tag_node_instance{instan}']=1
            else:
                feats[f'm_tag_node_instance{instan}']=0
        ##  node_cpu2
        node_cgst=metric_node[metric_node['tag_metric_name']=='node_cpu_guest_seconds_total']
        node_cgst_exit_mode=node_cgst['tag_mode'].unique()
        feats['m_tag_node_cgst_mode_n']=len(node_cgst_exit_mode)
        for mode in ['nice', 'user']:
            if mode in node_cgst_exit_mode:
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_node_cpu_guest_seconds_total_mode{mode}_v_{stats_func}']=node_cgst[node_cgst['tag_mode']==mode]['value'].agg(stats_func)         
            else:
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_node_cpu_guest_seconds_total_mode{mode}_v_{stats_func}']=-1         
        node_cst=metric_node[metric_node['tag_metric_name']=='node_cpu_seconds_total']
        node_cst_exit_mode=node_cst['tag_mode'].unique()
        feats['m_tag_node_cst_mode_n']=len(node_cst_exit_mode)
        for mode in ['nice', 'user', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'system']:
            if mode in node_cst_exit_mode:
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_node_cpu_seconds_total_mode{mode}_v_{stats_func}']=node_cst[node_cst['tag_mode']==mode]['value'].agg(stats_func)         
            else:
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_node_cpu_seconds_total_mode{mode}_v_{stats_func}']=-1                 
        ## node_94
        for n in metric_tags_94:
            if n in metric_exit_tag_name:
                temp=metric_node[metric_node['tag_metric_name']==n]
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_{n}_v_{stats_func}']=temp['value'].agg(stats_func) 
            else:
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_{n}_v_{stats_func}']=-1

        #  other
        for t in metric_ts:
            if t in metric_exit_timestamp:
                temp=metric[metric['timestamp']==t]
                feats[f'm_ts{t}_len']=len(temp)
                feats[f'm_ts{t}_dict_len']=temp['tags_dict_len'].mean()
            else:
                feats[f'm_ts{t}_len']= -1
                feats[f'm_ts{t}_dict_len']= -1              
        for stats_func in ['mean','max','min','ptp','nunique']:
            feats[f"m_tag_dict_len_{stats_func}"] = metric['tags_dict_len'].agg(stats_func) 
        for stats_func in ['max','min','nunique']:
            feats[f"m_ts_{stats_func}"] = metric['timestamp'].agg(stats_func) 
        s_metric=metric.groupby(['tag_metric_name'])[['value']].agg('mean').reset_index()
        for stats_func in ['mean','std','max','min','ptp','nunique']:
            feats[f"m_s_values_{stats_func}"] = s_metric['value'].agg(stats_func)

        #  add new node
        for mode in ['nice', 'user']:
            if mode in node_cgst_exit_mode:
                for job in metric_node_jobs3:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_nodenew_cpu_guest_seconds_total_job{job}_mode{mode}_v_{stats_func}']=node_cgst[(node_cgst['tag_mode']==mode)&(node_cgst['tag_job']==job)]['value'].agg(stats_func)         
            else:
                for job in metric_node_jobs3:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_nodenew_cpu_guest_seconds_total_job{job}_mode{mode}_v_{stats_func}']=-1    
        for mode in ['nice', 'user', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'system']:
            if mode in node_cst_exit_mode:
                for job in metric_node_jobs3:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_nodenew_cpu_seconds_total_job{job}_mode{mode}_v_{stats_func}']=node_cst[(node_cst['tag_mode']==mode)&(node_cst['tag_job']==job)]['value'].agg(stats_func)         
            else:
                for job in metric_node_jobs3:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_nodenew_cpu_seconds_total_job{job}_mode{mode}_v_{stats_func}']=-1        
        for n in metric_tags_94:
            if n in metric_exit_tag_name:
                temp=metric_node[metric_node['tag_metric_name']==n]
                for job in metric_node_jobs3:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_nodenew_{n}_job{job}_v_{stats_func}']=temp[temp['tag_job']==job]['value'].agg(stats_func) 
            else:
                for job in metric_node_jobs3:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_nodenew_{n}_job{job}_v_{stats_func}']=-1


        del metric

    else:    #  2+34*2+ 1280 + 253 + 7680 +1932 +6160  +524 +1248=19147
        feats['m_l']=-10
        feats['m_l_nodup'] =-10

        for n in metric_tags_keys:
            feats[f'm_tag_{n}_n']=-10
            feats[f'm_tag_{n}_c']=-10

        # metric_service_4                           # 4*40*(8)=1280
        for n in metric_tags_4:
            for sn in metric_tag_service_name_test:
                feats[f'm_tag_{n}_service{sn}_ts_len']=-10
                for stats_func in ['mean','min','max','ptp','median']:
                    feats[f'm_tag_{n}_service{sn}_v_{stats_func}']=-10
                feats[f'm_tag_{n}_service{sn}_v_meanmin']=-10
                feats[f'm_tag_{n}_service{sn}_v_maxmean']=-10                              

        # metric_container15                          # 1+4*3+80+80*2=253
        feats['m_tag_container_job_n']=-10
        for ins in metric_tag_instance_nsdefault:
            feats[f'm_tag_instance{ins}_len']=-10
            feats[f'm_tag_instance{ins}_pod_n']=-10      
            feats[f'm_tag_instance{ins}_container_n']=-10             
        for ct in metric_container_80:
            feats[f'm_tag_contain_contain{ct}_len']=-10
        for sn in metric_pod80:
            feats[f'm_pod{sn}_nodeinstance']=-10
            feats[f'm_pod{sn}_ts_len']=-10 

        for n in metric_tag_contain_network8:             #  8*(31*16+116*4) =7680
            for sn in metric_tag_net8_4instance_pod7:           # 31
                for interface in metric_contain_net8_pod2interface[f'{sn}']:
                    for ins in metric_tag_instance_nsdefault:
                        for stats_func in ['mean','min','max','ptp']:
                            feats[f'm_tag_{n}_pod{sn}_interface{interface}_ins{ins}_v_{stats_func}']=-10   
            for sn in metric_pod80:                             # 116
                for interface in metric_contain_net8_pod2interface[f'{sn}']:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_pod{sn}_interface{interface}_v_{stats_func}']=-10

        for n in metric_tag_cpu7_2container_3:             # 3*(((1+4*4)+ (53*5))+ 1+4*2*4+ 1+4*4+ 3*(1+8)+ 57*5)=1932
            ##  container ‘’ 
            for pod in metric_tag_cpu7_2container_3_empty_4instance1_pod1:
                feats[f'm_tag_{n}_pod{pod}_ts_0len']=-10
                for node in metric_tag_instance_nsdefault:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_pod{pod}_node{node}_v_{stats_func}']=-10
            for pod in metric_tag_cpu7_2container_3_emptyct_pod53:
                feats[f'm_tag_{n}_pod{pod}_ts_len']=-10
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_{n}_pod{pod}_v_{stats_func}']=-10
            ##  container not empty
            ###   pod='nacosdb-mysql'
            pod='nacosdb-mysql'
            feats[f'm_tag_{n}_notpod{pod}_ts_0len']=-10
            for node in metric_tag_instance_nsdefault:
                for ct in ['mysql','xenon']:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_notpod{pod}_node{node}_ct{ct}_v_{stats_func}']=-10
            ### pod=='kube-flannel'
            pod='kube-flannel'
            feats[f'm_tag_{n}_notpod{pod}_ts_1len']=-10
            for node in metric_tag_instance_nsdefault:
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_{n}_notpod{pod}_node{node}_v_{stats_func}']=-10
            ### pod in ['7f1cbe89dd024b6ebbf5556426b34acf', 'c019762f3cbd493cb4dc4443eec2c273', 'ab1c1a0046754e49b4fdf64708124365']
            for pod in metric_tag_cpu7_2container_3_notempty2ct_1instance_pod3:
                feats[f'm_tag_{n}_notpod{pod}_ts_2len']=-10
                for ct in ['mysql','xenon']:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_notpod{pod}_ct{ct}_v_{stats_func}']=-10
            ### other pod
            for pod in metric_tag_cpu7_2container_3_notemptyct_pod57:
                feats[f'm_tag_{n}_notpod{pod}_ts_len']=-10
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_{n}_notpod{pod}_v_{stats_func}']=-10

        for n in metric_tag_cpu7_3container_4:             # 4*( 7*(1+16)+73*5+ 7*(1+16)+73*5+ 1+4*3*4+ 3*(1+12)+ 7*(1+16)+ 73*5)=6160
            ##  container ‘’ 
            for pod in metric_tag_cpu7_3container_4_4_instance7:
                feats[f'm_tag_{n}_pod{pod}_ts_len']=-10
                for node in metric_tag_instance_nsdefault:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_pod{pod}_node{node}_v_{stats_func}']=-10 
            for pod in metric_contain_cpu7_instance73:
                feats[f'm_tag_{n}_pod{pod}_ts_len']=-10
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_{n}_pod{pod}_v_{stats_func}']=-10
            ##  container POD
            for pod in metric_tag_cpu7_3container_4_4_instance7:
                feats[f'm_tag_{n}_podpod{pod}_ts_len']=-10
                for node in metric_tag_instance_nsdefault:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_podpod{pod}_node{node}_v_{stats_func}']=-10 
            for pod in metric_contain_cpu7_instance73:
                feats[f'm_tag_{n}_podpod{pod}_ts_len']=-10
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_{n}_podpod{pod}_v_{stats_func}']=-10
            ##  container other 
            ###  pod='nacosdb-mysql'
            pod='nacosdb-mysql'
            feats[f'm_tag_{n}_notpod{pod}_ts_0len']=-10
            for node in metric_tag_instance_nsdefault:
                for ct in ['mysql','xenon','slowlog']:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_notpod{pod}_node{node}_ct{ct}_v_{stats_func}']=-10 
            ### pod in ['7f1cbe89dd024b6ebbf5556426b34acf', 'c019762f3cbd493cb4dc4443eec2c273', 'ab1c1a0046754e49b4fdf64708124365']
            for pod in metric_tag_cpu7_2container_3_notempty2ct_1instance_pod3:
                feats[f'm_tag_{n}_notpod{pod}_ts_1len']=-10
                for ct in ['mysql','xenon','slowlog']:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_notpod{pod}_ct{ct}_v_{stats_func}']=-10 
            ### pod in metric_tag_cpu7_3container_4_4_instance7
            for pod in metric_tag_cpu7_3container_4_4_instance7:
                feats[f'm_tag_{n}_notpod{pod}_ts_len']=-10
                for node in metric_tag_instance_nsdefault:
                    for stats_func in ['mean','min','max','ptp']:
                        feats[f'm_tag_{n}_notpod{pod}_node{node}_v_{stats_func}']=-10
            ### other pod
            for pod in metric_contain_cpu7_instance73:
                feats[f'm_tag_{n}_notpod{pod}_ts_len']=-10
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_{n}_notpod{pod}_v_{stats_func}']=-10 

        # metric_node94                      3+3+44+1+2*4+1+8*4+94*4+21*2+5+3+6=524
        feats[f'm_tag_node_instance_n']=-10
        feats[f'm_tag_node_job_n']=-10
        feats[f'm_tag_node_kubernetes_pod_name_n']=-10
        for job in metric_node_jobs3:
            feats[f'm_tag_node_job{job}']=-10
        for instan in metric_tag_node_instance44:
            feats[f'm_tag_node_instance{instan}']=-10
        ##  node_cpu2                   
        feats['m_tag_node_cgst_mode_n']=-10
        for mode in ['nice', 'user']:
            for stats_func in ['mean','min','max','ptp']:
                feats[f'm_tag_node_cpu_guest_seconds_total_mode{mode}_v_{stats_func}']=-10  
        feats['m_tag_node_cst_mode_n']=-10
        for mode in ['nice', 'user', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'system']:
            for stats_func in ['mean','min','max','ptp']:
                feats[f'm_tag_node_cpu_seconds_total_mode{mode}_v_{stats_func}']=-10
        ## node_94
        for n in metric_tags_94:
            for stats_func in ['mean','min','max','ptp']:
                feats[f'm_tag_{n}_v_{stats_func}']=-10

        #  other
        for t in metric_ts:
            feats[f'm_ts{t}_len']= -10
            feats[f'm_ts{t}_dict_len']= -10       
        for stats_func in ['mean','max','min','ptp','nunique']:
            feats[f"m_tag_dict_len_{stats_func}"] =-10
        for stats_func in ['max','min','nunique']:
            feats[f"m_ts_{stats_func}"] =-10
        for stats_func in ['mean','std','max','min','ptp','nunique']:
            feats[f"m_s_values_{stats_func}"] =-10

        #  newnode            1248
        for mode in ['nice', 'user']:
            for job in metric_node_jobs3:
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_nodenew_cpu_guest_seconds_total_job{job}_mode{mode}_v_{stats_func}']=-1         
        for mode in ['nice', 'user', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'system']:
            for job in metric_node_jobs3:
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_nodenew_cpu_seconds_total_job{job}_mode{mode}_v_{stats_func}']=-1                 
        ## node_94
        for n in metric_tags_94:
            for job in metric_node_jobs3:
                for stats_func in ['mean','min','max','ptp']:
                    feats[f'm_tag_nodenew_{n}_job{job}_v_{stats_func}']=-10
    return feats


def get_feature():
    train_feature=[]
    for i in range(5):
        ds_num=i+1
        train_ids_i= set([n.split("_")[0] for n in os.listdir(f"data/{ds_num}/metric/")])|\
                        set([n.split("_")[0] for n in os.listdir(f"data/{ds_num}/log/")])|\
                        set([n.split("_")[0] for n in os.listdir(f"data/{ds_num}/trace/")])
        train_ids_i= list(train_ids_i)
        train_ids_i.sort()                             ### 增加列表排序功能
        print(f'dataset_{ds_num} ids: {len(train_ids_i)}')
        train_feature_i = pd.DataFrame(Parallel(n_jobs=-1,backend="multiprocessing")(delayed(processing_feature)(f,ds_num) for f in tqdm(train_ids_i)))
        train_feature.append(train_feature_i)
    train_feature_df=pd.concat(train_feature)
    test_ids= set([n.split("_")[0] for n in os.listdir(f"data/0/metric/")])|\
                    set([n.split("_")[0] for n in os.listdir(f"data/0/log/")])|\
                    set([n.split("_")[0] for n in os.listdir(f"data/0/trace/")])
    print(f'dataset_0 ids: {len(test_ids)}')
    test_ids=list(test_ids)
    test_ids.sort()                                      ### 增加列表排序功能
    test_feature_df = pd.DataFrame(Parallel(n_jobs=-1, backend="multiprocessing")(delayed(processing_feature)(f,0) for f in tqdm(test_ids) ))
    print(train_feature_df.info(),test_feature_df.info(),)
    return train_feature_df,test_feature_df
tr_f_df,te_f_df=get_feature()

# select the features
original_f=tr_f_df.columns
print('number of oringin features(with id):',len(original_f))
drop_f=[]
save_f=[]
for mode in ['nice', 'user']:
    for job in metric_node_jobs3:
        for stats_func in ['mean','min','max','ptp']:
            save_f.append(f'm_tag_nodenew_cpu_guest_seconds_total_job{job}_mode{mode}_v_{stats_func}')         
for mode in ['nice', 'user', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'system']:
    for job in metric_node_jobs3:
        for stats_func in ['mean','min','max','ptp']:
            save_f.append(f'm_tag_nodenew_cpu_seconds_total_job{job}_mode{mode}_v_{stats_func}')                 
for n in metric_tags_94:
    for job in metric_node_jobs3:
        for stats_func in ['mean','min','max','ptp']:
            save_f.append(f'm_tag_nodenew_{n}_job{job}_v_{stats_func}')
for i in set(tr_f_df.columns)-set(save_f):
    if len(set(tr_f_df[i].value_counts().index)-set([-1,-2,-3,-10]))<=1:
        drop_f.append(i)
print('number of drop features(with id):',len(drop_f))                      #drop 5888  features

id_feature_name = [i for i in original_f if i not in drop_f]
print(f"number of  reserve features(with id) = {len(id_feature_name)}")     #reserve 13259 features

tr_df= tr_f_df[id_feature_name]
tr_df.to_csv('model/tmp_data/ex_f_train_feature.csv',index=False)
te_df= te_f_df[id_feature_name]
te_df.to_csv('model/tmp_data/ex_f_test_feature.csv',index=False)

print('train and test features info:','\n',tr_df.info(),te_df.info())

print('\n','******** feature extract successfully ********','\n')