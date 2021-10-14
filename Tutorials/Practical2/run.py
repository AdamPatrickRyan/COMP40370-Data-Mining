import pandas as pd
import sklearn as sk
from sklearn import preprocessing,decomposition
import matplotlib.pyplot as plt
import os


def cleanse_missing_grades(df):
    """A function to replace the missing grades for HW and Exam with 0"""

    for hwno in range(1,4):
        df['Homework {}'.format(hwno)]=df['Homework {}'.format(hwno)].fillna(0)
    
    df['Exam']=df['Exam'].fillna(0)
    
def question_one_part_one(df):
    """A function to complete Q1P1"""
    
    print("""
    -
    QUESTION 1 - P1
    -
        
        
    """)
    
    def df_describe(df):
        """The dataframe before cleansing"""
        
        answer_df=df.describe().T

        print("The df shape is {}".format(df.shape))
        mean_col_name='mean'
        min_col_name='min'
        max_col_name='max'
        std_col_name='std'
        count_col_name='count'



        for homework_no in range(1,4):
            homework_column="Homework {}".format(homework_no)
            answer_statement="""
    Homework: {}
    Minimum: {}
    Maximum: {}
    Mean: {}
    Std. Deviation: {}
    Count: {}
    """
            filter_df=answer_df[answer_df.index==homework_column]

            min_col=filter_df[min_col_name][0]
            max_col=filter_df[max_col_name][0]
            mean_col=filter_df[mean_col_name][0]
            std_col=filter_df[std_col_name][0]
            cnt_col=filter_df[count_col_name][0]

            print(answer_statement.format(homework_column,
                                          min_col,
                                          max_col,
                                          mean_col,
                                          std_col,
                                         cnt_col))

        return
    
    df=pd.read_csv("./specs/Students_Results.csv")
    
    print("""
    ---
    Before Cleansing
    ---

    """)
    
    df_describe(df=df)
    
    cleanse_missing_grades(df=df)
    
    print("""
    ---
    After Cleansing
    ---

    """)
    
    
    df_describe(df=df)

    return df

def question_one_part_two(df):
    
    print("""
    -
    QUESTION 1 - P2
    -
        
        
    """)
    
    df['Homework Avg']=round((
                            (  (df['Homework 1'] + 
                                df['Homework 2'] +
                                df['Homework 3'])/3
                            )
                       ),1)
    
    df['Overall Mark']=round(0.25*df['Homework Avg'] + 0.75*df['Exam'],2)
    
    return df

def question_one_part_three(df):
    """Create a correlation matrix"""
    
    print("""
    -
    QUESTION 1 - P3
    -
    """)
    
    print("Correlation matrix: ")
    
    non_stud_col=[]
    
    for col in df.columns:
        
        if col!='Student ID':
            non_stud_col+=[col]
    
    corr_df=df[non_stud_col].corr()
    
    print(corr_df)

    return

def question_one_part_four():
    """See assignment"""
    return

def question_one_part_five(df):
    """Question 1 Part 5"""
    
    def ucd_grading(row):
        """UCD Computer Science Grading"""
        
        Ap=95
        A=90
        Am=85
        Bp=80
        B=75
        Bm=70
        Cp=65
        C=60
        Cm=55
        Dp=50
        D=45
        Dm=40
        Ep=35
        E=30
        Em=25
        Fp=20
        F=15
        Fm=10
        Gp=8
        G=5
        Gm=2
        NG=0
        
        if row['Overall Mark']>=Ap and row['Overall Mark']<=100:
              return 'A+'
            
        if row['Overall Mark']>=A and row['Overall Mark']<Ap:
              return 'A'
            
        if row['Overall Mark']>=Am and row['Overall Mark']<A:
              return 'A-'
            
            
            
        if row['Overall Mark']>=Bp and row['Overall Mark']<Am:
              return 'B+'
            
        if row['Overall Mark']>=B and row['Overall Mark']<Bp:
              return 'B'
            
        if row['Overall Mark']>=Bm and row['Overall Mark']<B:
              return 'B-'
            
            
        if row['Overall Mark']>=Cp and row['Overall Mark']<Bm:
              return 'C+'
            
        if row['Overall Mark']>=C and row['Overall Mark']<Cp:
              return 'C'
            
        if row['Overall Mark']>=Cm and row['Overall Mark']<C:
              return 'C-'
            
            
            
        if row['Overall Mark']>=Dp and row['Overall Mark']<Cm:
              return 'D+'
            
        if row['Overall Mark']>=D and row['Overall Mark']<Dp:
              return 'D'
            
        if row['Overall Mark']>=Dm and row['Overall Mark']<D:
              return 'D-'
            
            
            
        if row['Overall Mark']>=Ep and row['Overall Mark']<Dm:
              return 'E+'
            
        if row['Overall Mark']>=E and row['Overall Mark']<Ep:
              return 'E'
            
        if row['Overall Mark']>=Em and row['Overall Mark']<E:
              return 'E-'
            

            
        if row['Overall Mark']>=Fp and row['Overall Mark']<Em:
              return 'F+'
            
        if row['Overall Mark']>=F and row['Overall Mark']<Fp:
              return 'F'
            
        if row['Overall Mark']>=Fm and row['Overall Mark']<F:
              return 'F-'
            
            
        if row['Overall Mark']>=Gp and row['Overall Mark']<Bm:
              return 'G+'
            
        if row['Overall Mark']>=G and row['Overall Mark']<Gp:
              return 'G'
            
        if row['Overall Mark']>=Gm and row['Overall Mark']<G:
              return 'G-'
            
        if row['Overall Mark']>=NG and row['Overall Mark']<Gm:
              return 'NG'
            
        return 'Grade Error'
        
    df['Grade']= df.apply(lambda row: ucd_grading(row), axis=1)
    
    ax=(df['Overall Mark']
            .plot(
                kind='hist'
                ,title='Histogram of Overall Marks'
                ,xlabel='Overall Mark'
                ,ylabel='No of Students'
                ,bins=10
                )
            .set_xlim((0,100))
        )
    
    
    
    plt.savefig('./output/histogram_of_grades_q1.png')
    
    
    
    df_grade=df['Grade'].value_counts().reindex(index = ['A+','A','A-',
                                                         'B+','B','B-',
                                                         'C+','C','C-',
                                                         'D+','D','D-',
                                                        'E+','E','E-',
                                                        'F+','F','F-',
                                                        'G+','G','G-'])
    
    new_ax=(df_grade
            .plot(
                kind='bar'
                ,title='Barchart of Grades'
                ,xlabel='Grade'
                ,ylabel='No of Students'
                )
        )
    
    plt.savefig('./output/barchart_of_grades_q1.png')
    
    
    return df

def question_one_part_six(df):
    df.to_csv('./output/question1_out.csv',index=False)

def question_one():
    student_df=pd.read_csv('./specs/Students_Results.csv')
    
    #Part 1
    student_df=question_one_part_one(df=student_df)
    
    #Part 2
    student_df=question_one_part_two(df=student_df)
    
    question_one_part_three(df=student_df)
    
    question_one_part_four()
    
    student_df=question_one_part_five(df=student_df)
    
    question_one_part_six(df=student_df)
    
    return student_df



def question_two_part_one(df):
    df['Original Input3']=df['Input3']
    df['Original Input12']=df['Input12']
    return df

def question_two_part_two(df):
    """Scale the columns"""
    
    #Z score
    df['Input3']=(df['Input3'] - df['Input3'].mean())/df['Input3'].std()

    return df
    
def question_two_part_three(df):
    """Scale the columns"""    

    #MinMax
    if df['Input12'].max()-df['Input12'].min()>0:
        #Have to round because of the test_practical only has three decimal places.
        df['Input12']=round((df['Input12']-df['Input12'].min())/(df['Input12'].max()-df['Input12'].min()),3)
    else:
        print("Error: Min=Max")
    
    return df

def question_two_part_four(df):
    
    inputs=[]
    for input_no in range(1,13):
        inputs+=['Input{}'.format(input_no)]
    
    df['Average Input']=df[inputs].mean(axis=1)
    
    return df

def question_two_part_five(df):
    
    df.to_csv('./output/question2_out.csv',index=False)
    
    return

def question_two():
    sensor_df=pd.read_csv('./specs/Sensor_Data.csv')
    sensor_df=question_two_part_one(sensor_df)
    sensor_df=question_two_part_two(sensor_df)
    sensor_df=question_two_part_three(sensor_df)
    sensor_df=question_two_part_four(sensor_df)
    question_two_part_five(sensor_df)
    return



def question_three_part_one(df):

    def pca_given_limit(df,cut_off_limit=0.95):
        """PCA to get components given a cutoff"""

        component_no=1
        explained_variance = {}
        explained_variance['default'] = -1

        #Add in components
        while max(explained_variance.values())<cut_off_limit and component_no<len(df.columns):
            pca = decomposition.PCA(n_components = component_no)
            pca.fit(df)
            post_pca_data = pca.transform(df)

            for component in range(0, len(pca.explained_variance_ratio_)):

                current_feature='PCA_{}'.format(component)

                if component == 0:
                    explained_variance[current_feature]=pca.explained_variance_ratio_[component]

                else:
                    previous_explained_variance=explained_variance['PCA_{}'.format(component-1)]
                    explained_variance[current_feature] = pca.explained_variance_ratio_[component] + previous_explained_variance

            component_no+=1



        return post_pca_data, explained_variance
    
    pca_data, var_dict = pca_given_limit(df,cut_off_limit=0.95)
    
    pca_col_names=[]
    
    for k in range(0,pca_data.shape[1]):
        pca_col_names+=['pca{}'.format(k)]
        
    pca_df=pd.DataFrame(pca_data)
    pca_df.columns=pca_col_names
        
    return pca_df, var_dict



def question_three_part_two(original_df,pca_df):
    """Use cut to complete part 2"""
    
    pca_columns=pca_df.columns
    
    for column in pca_columns:
        original_df["{}_width".format(column)]=pd.cut(pca_df[column],10)
        
    return original_df

def question_three_part_three(original_df,pca_df):
    """Use qcut to complete part three"""
    
    pca_columns=pca_df.columns
    
    for column in pca_columns:
        original_df["{}_freq".format(column)]=pd.qcut(pca_df[column],10)
        
    return original_df

def question_three_part_four(df):
    """Complete part 4 by saving the DF"""
    
    df.to_csv('./output/question3_out.csv',index=False)
        
    return

def question_three():
    """Question Three"""
    df=pd.read_csv('./specs/DNA_Data.csv')
    pca_df, var_dict=question_three_part_one(df)
    question_three_part_two(original_df=df,pca_df=pca_df)
    question_three_part_three(original_df=df,pca_df=pca_df)
    question_three_part_four(df=df)
    return
    


def run_all():
    """A function to run all of the tutorial"""

    if not os.path.exists('./output/'):
        os.makedirs('./output/')

    question_one()
    question_two()
    question_three()

run_all()