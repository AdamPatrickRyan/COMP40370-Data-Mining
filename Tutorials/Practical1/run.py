import pandas as pd
import numpy as np
import os



def my_mean(dataset=[1]):
    """A custom function to calculate a mean"""
    if type(dataset)!=list or len(dataset)==0:
        raise ValueError("Invalid")
    
    total=0
    
    for x in dataset:
        total+=x
        
    mean=total/len(dataset)
    
    return mean

def my_median(dataset=[1]):
    """A custom function to calculate a median"""
    
    if type(dataset)!=list or len(dataset)==0:
        raise ValueError("Invalid")
    
    for value in dataset:
        if type(value)!=float and type(value)!=int:
            raise ValueError("Should be list of value")
    
    dataset.sort()
    length_of_dataset=len(dataset)
    
    if length_of_dataset%2==1:
        median_index=length_of_dataset//2 #+ 1 <- Python is zero indexed
        print('Median Index: ',median_index)
        median_value=dataset[median_index]
        
        
    else:
        median_index=length_of_dataset//2
        print('Median Index: ',median_index)
        median_value=(dataset[median_index-1] + dataset[median_index]) / 2 #Python is zero indexed
        
    return median_value
    
def my_mode(dataset=[1]):
    """A custom function to get the mode"""
    if type(dataset)!=list or len(dataset)==0:
        raise ValueError("Invalid")
    
    for value in dataset:
        if type(value)!=float and type(value)!=int:
            raise ValueError("Should be list of value")
    
    dict_of_values={}
    
    for value in dataset:
        if value not in dict_of_values:
            dict_of_values[value]=0
        else:
            dict_of_values[value]+=1
    
    modal_count=max(dict_of_values.values())
    
    mode_count=0

    for key,value in dict_of_values.items():
        
        if value == modal_count:
            mode_count+=1
            print("Mode {} is: {}".format(mode_count,key))

    return max(dict_of_values, key=dict_of_values.get)
    
    
def three_question_one(data_set):
    """This is the solution to question three part 1 in the tutorial sheet"""
    
    question_one_string="""
The mean: {}
The median: {}
"""

    print(question_one_string.format(np.mean(data_set),np.median(data_set),np.std(data_set)))
    
    return 
    
def three_question_two(data_set):
    """This is the solution to question three part 2 in the tutorial sheet"""
    
    my_mode_value=my_mode(dataset=data_set)
    
    question_two_string="""
What is the mode of the data? 
Answer: {} 
    """

    print(question_two_string.format(my_mode_value))

    answer_two_part_two_string="""
The data’s modality is: {}
    """
    print(answer_two_part_two_string.format('Bimodal, skewed right - we see two values appear the modal number of times: 25 and 35'))
    
    return
          
   
def three_question_three(data_set):
    """This is the solution to question three p 3 in the tutorial sheet"""
    
    max_data_set_value=np.max(data_set)
    min_data_set_value=np.min(data_set)
    mid_range=(max_data_set_value-min_data_set_value)/2
          
    answer_three_string="""
The mid-range of the data is: {}
"""
    print(answer_three_string.format(mid_range))

    return mid_range

def three_question_four(data_set):
    """This is the solution to Q3p4"""
    
    first_quartile=np.percentile(data_set, [25])[0]
    third_quetile=np.percentile(data_set, [75])[0]
    
    print("""
Q1: {}
Q3: {}
""".format(np.percentile(data_set, [25])[0]
          ,np.percentile(data_set, [75])[0]))
    
    return

def three_question_five(data_set):
    """This is the solution to Q3p5"""
    
    def five_number_summary(list_of_numbers=[1]):
        """A function to get five_number_summary as described in Lecture 2 Slide 21"""
    
        if type(list_of_numbers)!=list:
            raise ValueError("Needs to be a list")

        value_dict={
            'min':np.min(list_of_numbers)
            ,'Q1':np.percentile(list_of_numbers, [25])[0]
            ,'median':np.median(list_of_numbers) 
            ,'Q3':np.percentile(list_of_numbers, [75])[0]
            ,'max':np.max(list_of_numbers)

        }

        print("""
min()={} 
Q1={} 
median={} 
Q3={} 
max()={}
""".format(
            value_dict['min']
            ,value_dict['Q1']
            ,value_dict['median']
            ,value_dict['Q3']
            ,value_dict['max']
                        )
             )

        return value_dict

    five_number_summary(list_of_numbers=data_set)
    
    return



def three_question_six(data_set):
    """This is the solution to Q3p6"""
    
    age_df=pd.DataFrame(data_set)
    
    image=(
            age_df
               .plot
               .box(title="A Barplot of Ages for Q3P6")
                   
          )
    
    image.figure.savefig('Q3P6_Barplot.png')
    
    return

def question_three():
    """A function to run question 3"""
    
    data_set=[13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]
    three_question_one(data_set)
    three_question_two(data_set)
    three_question_three(data_set)
    three_question_four(data_set)
    three_question_five(data_set)
    three_question_six(data_set)

    return


def question_four(file_path='./specs/AutoMpg_question1.csv'):
    """This documents the solution to question 4 - data cleansing"""
    
    df=pd.read_csv(file_path)
    
    if 'horsepower' not in df or 'origin' not in df:
        raise ValueError("""Needs to contain horsepower and origin""")

    print("""There were {} missing horsepower values and {} missing origin values""".format(len(df[df['horsepower'].isna()]),len(df[df['origin'].isna()])))

    df['horsepower']=df['horsepower'].fillna(df['horsepower'].mean())
    df['origin']=df['origin'].fillna(df['origin'].min())
    
    df.to_csv('./output/question1_out.csv',index=False)
        
    return


def question_five(file_path_a="./specs/AutoMpg_question2_a.csv",file_path_b="./specs/AutoMpg_question2_b.csv"):
    """This documents the solution to question 5 - data integration"""
    
    df_a=pd.read_csv(file_path_a)
    df_b=pd.read_csv(file_path_b)
    
    if 'car name' not in df_a or 'name' not in df_b or 'other' not in df_b:
        raise ValueError("""Does not have the expected columns""")
        
    df_b=df_b.rename(columns={'name':'car name'})
    df_a['other']=1
    
    concatenated_frames=pd.concat([df_a,df_b])
        
    concatenated_frames.to_csv('./output/question2_out.csv',index=False)
    
    return

def run_all():
    """A function to run all of the tutorial"""

    if not os.path.exists('./output/'):
        os.makedirs('./output/')

    question_three()
    question_four()
    question_five()

run_all()