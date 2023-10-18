import streamlit as st
import time
from ucimlrepo import fetch_ucirepo
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import altair as alt

st.set_option('deprecation.showPyplotGlobalUse', False)

heart_disease = fetch_ucirepo(id=45)
df = heart_disease.data.original
df['num_group'] = (df['num'] > 0).astype(int)

df.dropna(axis=0, inplace=True)

def diag(num):
    if num == 0:
        return 'Not diagnosed'
    elif num == 1:
        return 'Diagnosed'
    else:
        return 'Unknown'

def gen(num):
    if num == 0:
        return 'Female'
    elif num == 1:
        return 'Male'
    else:
        return 'Unknown'

df['diagnosis'] = df['num_group'].apply(diag)
df['gender'] = df['sex'].apply(gen)

df = df.drop(columns=['sex', 'num', 'num_group'])

st.markdown('# Heart Disease')
st.image('https://buzzrx.s3.amazonaws.com/1ac6c8f4-059d-44b3-81fa-6827559ed545/6CausesofHeartDisease.png')

st.markdown('In the United States, **cardiovascular diseases** rank as the primary cause of mortality. This means that, on average, \
someone succumbs to a cardiovascular disease every 36 seconds. **Heart disease**, in particular, leads to the demise of over 650,000 individuals \
annually in the U.S., equating to _1 out of every 4 deaths_. The economic impact of heart disease is substantial, \
with the nation incurring more than **$350** billion annually in healthcare expenses, medication costs, and the financial toll of premature deaths.\
Globally, cardiovascular disease exacts a staggering toll, claiming the lives of nearly 18 million people each year, \
representing a staggering one-third of all recorded deaths.')
st.markdown('But you can lower your risk of heart disease to a large extent with behavioral and lifestyle changes. \
Hence, it is crucial to understand the potential risk factors contributing to heart disease.')
st.markdown('**UCI Heart Disease dataset** emerges as a highly promising resource for evaluating heart disease risk.')
st.markdown('This web app provides visualization of heart disease dataset, which is designed for users to perform targeted searches,\
gain profound insights,and engage in comprehensive data analysis customized to their individual health profiles.')

tab1, tab2, tab3, tab4= st.tabs(['Introduction', 'Age & Gender', 'Other risk factors', 'Correlation'])

with tab1:
    st.markdown('''
    **The heart disease dataset collected data from 303 consecutive patients referred for coronary angiography at the Cleveland Clinic \
    between May 1981 and September 1984.** None of the patients had a previous history of myocardial infarction or evidence of myocardial infarction \
    in their electrocardiograms. There was also no known history of valvular or cardiomyopathic diseases among these 303 patients. \
    During their routine evaluations, these patients provided their medical histories, underwent different examinations, such as physical examinations, resting electrocardiograms, \
    serum cholesterol measurements, and fasting blood sugar tests, etc.
    The results of these tests were not interpreted until after the coronary angiograms had been read. These tests were analyzed, and the results were \
    recorded without any knowledge of the historical or angiographic results. Therefore, _there was no work-up bias._
     the dataset does have some missing data, such as "ca" (number of major vessels, ranging from 0 to 3, colored by fluoroscopy) and \
     "thal" (thalassemia), these missing data constitute less than 1% of the overall dataset and bear no significant impact on subsequent data analysis. \
     Therefore, we do
     The missingness of this dataset is missing completely at random (MCAR).
    ''')

    with st.expander("Summary Statistics:"):
        st.write(df.describe())

    count_female = df[df['gender'].str.contains('Female')].shape[0]
    count_male = df[df['gender'].str.contains('Male')].shape[0]

    #per_famale = count_female/(count_male+count_female)
    #per_male = count_male/(count_male+count_female)

    count_diagnosed = df[df['diagnosis'].str.contains('Diagnosed')].shape[0]
    count_un = df[df['diagnosis'].str.contains('Not diagnosed')].shape[0]

    #per_diagnosed = count_diagnosed/(count_diagnosed+count_un)
    #per_un = ount_diagnosed/(count_diagnosed+count_un)

    gender_l = pd.DataFrame({"category": ['Female', 'Male'], "value": [count_female, count_male]})
    diagnosis_l = pd.DataFrame({"category": ['Diagnosed', 'Not Diagnosed'], "value": [count_diagnosed, count_un]})

    chart1 = alt.Chart(gender_l).mark_arc().encode(
    theta="value",
    color=alt.Color('category', legend=alt.Legend(title='Gender')),
    text=alt.Text('value')
    ).properties(
    height=300,
    width=300,
    title='Gender distribution'
    )
    chart1
    st.markdown('Among the participants, there are 201 males:man: and 96 females:woman:, accounting for 67.7% and 32.3% of the total data, respectively.')

    chart2 = alt.Chart(diagnosis_l).mark_arc().encode(
    theta="value",
    color=alt.Color('category', legend=alt.Legend(title='Diagnosis'), scale = alt.Scale(scheme ="tableau10")),
    text=alt.Text('value')
    ).properties(
    height=300,
    width=300,
    title='Diagnosis distribution'
    )
    chart2
    st.markdown('The dataset shows 137 participants are diagnosed with heart disease, while 161 participants are not diagnosed.')



#The average age of these patients was 54 years,\
#with 206 of them being men. The angiograms were interpreted by a cardiologist who had no knowledge of other test data. \
#Further details regarding this data collection are described elsewhere.


with tab2:
    st.markdown('**Gender** and **age** are indispensable considerations when assessing risk factors for heart disease, as they exert a significant \
    impact on disease prevalence and outcomes. Furthermore, these factors are readily observable. \
    Let us delve into some basic analyses to gain a deeper understanding of how gender and age affect the likelihood of heart disease diagnosis.')

    factor_list = ['gender', 'age']
    factor = st.radio('Choose the factors that you are interested in: age or gender', factor_list)

    if factor == 'gender':
        st.markdown(' ')
        st.markdown('### GENDER Differences')
        st.markdown('According to the analysis based on the dataset, we can find the rate of heart disease diagnosis is much \
        **higher in males than in females**. Epidemiological studies also indicate that, compared to men, \
        women can to some extent prevent the development of cardiovascular diseases before menopause. \
        Women have a lower incidence of cardiovascular disease compared to age-matched men and tend to develop cardiovascular diseases \
        about 10 years later than men.[1] Although postmenopausal women are at a higher risk of developing cardiovascular diseases \
        compared to premenopausal women, their incidence of cardiovascular diseases is still much lower when compared to age-matched \
        men who are far past menopause.[2] Research findings suggest that the primary circulating female hormone, estrogen (E2), has a cardioprotective effect.[3]')

        total = len(df)
        ax = sns.countplot(data=df, x='gender', hue='diagnosis', palette='Set2')
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 1, f'{100 * height / total:.2f}%', ha="center", size = '10')
        st.pyplot()

        st.markdown('#### _References_')
        st.markdown('1. Wake R, Yoshiyama M. Gender differences in ischemic heart disease. Recent Patents Cardiovasc Drug Discov. 2009;4:234–40.')
        st.markdown('2. Arnold AP, Cassis LA, Eghbali M, Reue K, Sandberg K. Sex hormones and sex chromosomes cause sex differences \
        in the development of cardiovascular diseases. Arterioscler. Thromb. Vasc. Biol. 2017;ATVBAHA.116.307301.')
        st.markdown('3. Iorga, A., Cunningham, C.M., Moazeni, S. et al. The protective role of estrogen and estrogen receptors in \
        cardiovascular disease and the controversial use of estrogen therapy. Biol Sex Differ 8, 33 (2017).')


    if factor == 'age':
        st.markdown(' ')
        st.markdown('### Risks increase with AGE')
        st.markdown('According to the boxplot shown below, The **age** of individuals with heart disease is generally **higher** \
        compared to those without heart disease. It is because age plays a vital role in the deterioration of cardiovascular functionality, \
        resulting in an increased risk of cardiovascular disease (CVD) in older adults [1,2]. \
        The prevalence of CVD has also been shown to increase with age, in both men and women. The risks associated with CVD \
        increase with age, and these correspond to an overall decline in sex hormones, primarily of estrogen and testosterone.[3] ')

        ax = sns.catplot(data=df, x="diagnosis", y="age", hue = 'gender', kind="box")
        ax._legend.set_bbox_to_anchor((1.1, 0.5))
        st.pyplot()

        st.markdown('#### _References_')
        st.markdown('1.  Curtis A.B., Karki R., Hattoum A., Sharma U.C. Arrhythmias in Patients ≥ 80 Years of Age: Pathophysiology, Management, and Outcomes. J. Am. Coll. Cardiol. 2018;71:2041–2057.')
        st.markdown('2. North B.J., Sinclair D.A. The intersection between aging and cardiovascular disease. Circ. Res. 2012;110:1097–1108.')
        st.markdown('3. Rodgers JL, Jones J, Bolleddu SI, Vanthenapalli S, Rodgers LE, Shah K, Karia K, Panguluri SK. \
        Cardiovascular Risks Associated with Gender and Aging. J Cardiovasc Dev Dis. 2019 Apr 27;6(2):19.')




with tab3:
    st.markdown('### The distribution of diagnosis results based on different risk factors.')
    st.markdown('UCI Heart Disease dataset comprises a wealth of patient information,\
     spanning age, gender, cholesterol levels, blood pressure, and various other crucial indicators.')

    with st.expander("Click here to know more about the risk factors"):
        st.write("### _Here are some information of the possible risk factors associated with heart disease._")
        st.markdown('''
        1. Interestingly, most diagnosed individuals are **asymptomatic**, while among those without a diagnosis, there are more cases of angina (chest pain).'
        2. More diagnosed people have **high** resting blood pressure.
        3. People with heart disease tend to have **lower** maximum heart rates (thalach).
        4. A greater risk of heart disease among those who experience chest pain during exercise (exang).
        5. Almost no undiagnosed people exhibit **ST depression** induced by exercise relative to rest.
        6. Among diagnosed individuals, most have a **flat ST depression** induced by exercise relative to rest.
        7. People tend to have heart disease if they have major vessels **colored by flourosopy**.
        8. Individuals with **reversible thalassemia** tend to have a higher likelihood of heart disease.
        ''')


# Define a dictionary with options and corresponding plotting logic

    plot_options = {
        "cp": lambda: sns.histplot(data=df, x="cp", hue='diagnosis', element='step'),
        "trestbps": lambda: sns.histplot(data=df, x="trestbps", hue='diagnosis', element='step'),
        "chol": lambda: sns.histplot(data=df, x="chol", hue='diagnosis', element='step'),
        "fbs": lambda: sns.histplot(data=df, x="fbs", hue='diagnosis', element='step'),
        "restecg": lambda: sns.histplot(data=df, x="restecg", hue='diagnosis', element='step'),
        "thalach": lambda: sns.histplot(data=df, x="thalach", hue='diagnosis', element='step'),
        "exang": lambda: sns.histplot(data=df, x="exang", hue='diagnosis', element='step'),
        "oldpeak": lambda: sns.histplot(data=df, x="oldpeak", hue='diagnosis', element='step'),
        "slope": lambda: sns.histplot(data=df, x="slope", hue='diagnosis', element='step'),
        "ca": lambda: sns.histplot(data=df, x="ca", hue='diagnosis', element='step'),
        "thal": lambda: sns.histplot(data=df, x="thal", hue='diagnosis', element='step'),
        }

    selected_option = st.selectbox("Select the risk factor that you are interested in:", list(plot_options.keys()))

    if selected_option == 'cp':
        st.markdown('''
        **cp**: chest pain type
        - Value 1: typical angina
        - Value 2: atypical angina
        - Value 3: non-anginal pain
        - Value 4: asymptomatic
        ''')
    if selected_option == 'trestbps':
        st.markdown('''
        **trestbps**: resting blood pressure (in mm Hg on admission to the hospital)
        ''')
    if selected_option == 'chol':
        st.markdown('''
        **chol**: serum cholestoral in mg/dl
        ''')
    if selected_option == 'fbs':
        st.markdown('''
        **fbs**: fasting blood sugar > 120 mg/dl
        - 1 = true
        - 0 = false
        ''')
    if selected_option == 'restecg':
        st.markdown('''
        **restecg**: resting electrocardiographic results
        - Value 0: normal
        - Value 1: having ST-T wave abnormality (T wave inversions and/or ST \
        elevation or depression of > 0.05 mV)
        - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
        ''')
    if selected_option == 'thalach':
        st.markdown('''
        **thalach**: maximum heart rate achieved
        ''')
    if selected_option == 'exang':
        st.markdown('''
        **exang**: exercise induced angina
        - 1 = yes
        - 0 = no
        ''')
    if selected_option == 'oldpeak':
        st.markdown('''
        **oldpeak**: ST depression induced by exercise relative to rest
        ''')
    if selected_option == 'slope':
        st.markdown('''
        **slope**: the slope of the peak exercise ST segment
        - Value 1: upsloping
        - Value 2: flat
        - Value 3: downsloping
        ''')
    if selected_option == 'ca':
        st.markdown('''
        **ca:** number of major vessels (0-3) colored by flourosopy
        ''')
    if selected_option == 'thal':
        st.markdown('''
        **thal**: Thalassemia, an inherited blood disorder that causes your body to have less hemoglobin than normal.
        - 3 = normal
        - 6 = fixed defect
        - 7 = reversable defect
        ''')


# Check if the selected option exists in the dictionary
    if selected_option in plot_options:
    # Call the corresponding plotting function
        plot_function = plot_options[selected_option]
        plot_function()
        plt.xlabel(selected_option)
        plt.ylabel('Density')
        st.pyplot()
    else:
        st.write("Select a valid option.")


with tab4:
    st.markdown('### The relationship between two risk factors.')
    st.markdown('It is also important to know the relationship between risk factors.\
    Understanding the relationship could help us to prepare for the emerging symptoms.')

    col_list = [
                'age',
                'trestbps',
                'chol',
                'thalach',
                'oldpeak',
                'ca'
                ]

    st.markdown("**Select TWO risk factors that you are interested in:**")
    x_variable = st.selectbox("X Variable", col_list)
    y_variable = st.selectbox("Y Variable", col_list)

    option = [
    'gender',
    'diagnosis',
    'cp:O',
    'fbs:O',
    'restecg:O',
    'exang:O',
    'slope:O',
    'thal:O'
    ]
    hue_list = st.radio('Please choose the color characteristic that you would like to focus on', option, horizontal=True)
    # Create a scatter plot
    corr_chart = (
    alt.Chart(df).mark_circle().encode(
    x=x_variable,
    y=y_variable,
    color=alt.Color(hue_list,scale = alt.Scale(scheme ="tableau10"))
    ).properties(
    width=800,
    height=500
    ).interactive()
    )

    st.altair_chart(corr_chart)

    df_n = df.drop(columns=['gender',
    'diagnosis',
    'cp',
    'fbs',
    'restecg',
    'exang',
    'slope',
    'thal'])

    with st.expander("Correlation heatmap"):
        sns.heatmap(df_n.corr(),
            annot=True,
            annot_kws={"size": 10},
            vmin=-1,
            vmax=1,
            center=0,
            cmap='coolwarm')
        st.pyplot()
