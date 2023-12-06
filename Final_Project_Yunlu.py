import streamlit as st
import time
from ucimlrepo import fetch_ucirepo
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import altair as alt
import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



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

# define the feature data and target data
X = heart_disease.data.features
y = heart_disease.data.targets

# clean the data
# Find rows with missing data in X
missing_rows = X.isnull().any(axis=1)

# Drop corresponding rows from both X and y
X_clean = X.drop(missing_rows[missing_rows == True].index)
y_clean = y.drop(missing_rows[missing_rows == True].index)

y_clean['num_group'] = (y['num'] > 0).astype(int)
y_clean = y_clean.drop(columns = ['num'])

st.markdown('# Heart Disease')
st.image('https://buzzrx.s3.amazonaws.com/1ac6c8f4-059d-44b3-81fa-6827559ed545/6CausesofHeartDisease.png')

st.markdown('In the United States, **cardiovascular diseases** rank as the primary cause of mortality. This means that, on average, \
someone succumbs to a cardiovascular disease every 36 seconds.In particular, it leads to the death of over 650,000 individuals \
annually in the U.S., equating to _1 out of every 4 deaths_. The economic impact of heart disease is substantial, \
with the nation incurring more than **$350** billion annually in healthcare expenses, medication costs, and the financial toll of premature deaths.\
Globally, cardiovascular diseases cause the deaths of nearly 18 million people each year, accounting for about one-third of all global deaths.')
st.markdown('But you can lower your risk of heart disease to a large extent with behavioral and lifestyle changes. \
Hence, it is crucial to understand the potential risk factors contributing to heart disease.')
st.markdown('**UCI Heart Disease dataset** emerges as a highly promising resource for evaluating heart disease risk.')
st.markdown('This web app provides visualization of heart disease dataset, which is designed for users to perform targeted searches,\
gain profound insights,and engage in comprehensive data analysis customized to their individual health profiles.')

tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs(['Introduction', 'Age & Gender', 'Other risk factors', 'Correlation', 'Classification', 'Heart Disease Prediction', 'Conclusion & Meet the Developer'])

with tab1:
    st.markdown('''
    **The heart disease dataset collected data from 303 consecutive patients referred for coronary angiography at the Cleveland Clinic \
    between May 1981 and September 1984.**
    ''')

    st.markdown('''
    In a study of 303 patients, none had a history of myocardial infarction or signs of it in their electrocardiograms, nor did they \
    have any known valvular or cardiomyopathic diseases. During routine evaluations, they provided medical histories and underwent \
    various tests including physical exams, electrocardiograms, cholesterol and blood sugar tests. The test results were analyzed \
    independently, without prior knowledge of their medical histories or coronary angiogram results, ensuring no work-up bias.
    ''')

    st.markdown('''The dataset does have some missing data. These missing data constitute less than 1% of the overall dataset and bear no significant impact on subsequent data analysis. \
    The missingness of this dataset is _missing completely at random (MCAR)_. Therefore, the rows containing the missing data are removed.
    ''')

    with st.expander("Summary Statistics:"):
        st.write(df.describe())

    count_female = df[df['gender'].str.contains('Female')].shape[0]
    count_male = df[df['gender'].str.contains('Male')].shape[0]

    count_diagnosed = df[df['diagnosis'].str.contains('Diagnosed')].shape[0]
    count_un = df[df['diagnosis'].str.contains('Not diagnosed')].shape[0]


    gender_l = pd.DataFrame({"category": ['Female', 'Male'], "value": [count_female, count_male]})
    diagnosis_l = pd.DataFrame({"category": ['Diagnosed', 'Not Diagnosed'], "value": [count_diagnosed, count_un]})

    st.markdown('**The following is an overview of patient information extracted from the dataset.**')
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
    st.markdown('The dataset shows 137 participants are diagnosed with heart disease, while 160 participants are not diagnosed.')



#The average age of these patients was 54 years,\
#with 206 of them being men. The angiograms were interpreted by a cardiologist who had no knowledge of other test data. \
#Further details regarding this data collection are described elsewhere.


with tab2:
    st.markdown('**Gender** and **age** are important considerations when assessing risk factors for heart disease, as they exert a significant \
    impact on disease prevalence and outcomes. Furthermore, these factors are readily observable. \
    Let us delve into some basic analyses to gain a deeper understanding of how gender and age affect the probability of heart disease diagnosis.')

    factor_list = ['gender', 'age']
    factor = st.radio('**Choose the factors that you are interested in: age or gender**', factor_list)

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
    st.markdown('#### Explore Heart Disease Risk Factors')
    st.markdown('**Discover How Different Risk Factors Influence Heart Disease Diagnosis**')
    st.markdown('UCI Heart Disease dataset comprises a lot of patient information,\
     spanning age, gender, cholesterol levels, blood pressure, and various other crucial indicators.')
    st.markdown('In this section of the app, our interactive tool allows you to select a risk factor that interests you and \
    observe how it correlates with heart disease diagnoses.')

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
        **trestbps**: resting blood pressure (in mmHg)
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

    st.markdown('#### Discover Correlations Between Heart Disease Risk Factors')
    st.markdown('**Uncover the Relationship Between Different Health Indicators**')
    st.markdown('In this interactive module, you can explore the connections between various heart disease risk factors.\
    By selecting two different health indicators, you can visualize their correlation. \
    It will help in understanding how different factors might interact and influence each other in \
    the context of heart disease.')

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

with tab5:
    st.markdown('#### Explore Machine Learning Models')
    st.markdown('**An Interactive Way to Predict Heart Disease**')
    st.markdown('''Welcome to the heart disease prediction tool. In this section, you can \
    interact with different machine learning models and see how well they perform in \
    predicting the presence of heart disease based on clinical data.You can adjust the \
    hyperparameters for each model to see how they affect the model's accuracy.''')

    classifier_option = ['Random Forest Classifier', 'K-Nearest Neighbors', 'Support Vector Machine']

    cf_option = st.selectbox("Select the classifier tool that you want to use to predict:", classifier_option)

    if cf_option == 'Random Forest Classifier':
        st.image('https://serokell.io/files/vz/vz1f8191.Ensemble-of-decision-trees.png')

        test_fraction = st.slider('Select the proportion of the test data (%)', 0, 100, 20)

        st.markdown('**Please choose the hyperparameters you would like to set for the Random Forest algorithm:**')
        max_d = st.slider('Maximum Depth of Trees', 1, 10, 5)
        n_est = st.number_input("Number of Trees ", value=100)

        test_fraction = test_fraction/100

        start_state = 42

        #Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_fraction, random_state=start_state
            )

        # Define the Random Tree classifier
        classifier_rf = RandomForestClassifier(max_depth=max_d,
                                           n_estimators=n_est, oob_score=True)

        classifier_rf.fit(X_train, y_train)

        y_pred = classifier_rf.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        accu = (conf_mat[0,0]+conf_mat[1,1])/sum(sum(conf_mat))

        st.write(f'The accuracy of the algorithm is {accu:.2%}')

        st.write('Here is the confusion matrix')
        ConfusionMatrixDisplay.from_estimator(classifier_rf, X_test, y_test)
        st.pyplot()

    elif cf_option == 'K-Nearest Neighbors':
        st.image('https://miro.medium.com/v2/resize:fit:1010/format:webp/0*2_qzcm2gSe9l67aI.png')

        test_fraction = st.slider('Select the proportion of the test data (%)', 0, 100, 20)

        st.markdown('**Please choose the hyperparameters you would like to set for the Random Forest algorithm:**')
        n = st.slider('Number of neighbors to use', 1, 10, 5)

        al_option = ['auto', 'ball_tree', 'kd_tree', 'brute']
        al = st.selectbox("Select the algorithm that you want to use:", al_option)

        test_fraction = test_fraction/100

        start_state = 42


        #Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_fraction, random_state=start_state
            )

        # Define the Random Tree classifier
        classifier_knn = KNeighborsClassifier(n_neighbors=n, algorithm=al, leaf_size=30, p=2)

        classifier_knn.fit(X_train, y_train)

        y_pred = classifier_knn.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        accu = (conf_mat[0,0]+conf_mat[1,1])/sum(sum(conf_mat))

        st.write(f'The accuracy of the algorithm is {accu:.2%}')

        st.write('Here is the confusion matrix')
        ConfusionMatrixDisplay.from_estimator(classifier_knn, X_test, y_test)
        st.pyplot()

    elif cf_option == 'Support Vector Machine':
        st.image('https://editor.analyticsvidhya.com/uploads/97455support-vector-machine-algorithm5.png')

        test_fraction = st.slider('Select the proportion of the test data (%)', 0, 100, 20)

        st.markdown('**Please choose the hyperparameters you would like to set for the SVM algorithm:**')
        c = st.slider('Regularization parameter', 1, 5, 1)

        kernel_option = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        k = st.selectbox("Select the kernel that you want to use:", kernel_option)
        g = st.number_input("Kernel coefficient(gamma)", format="%.4f", value=0.001)

        test_fraction = test_fraction/100

        start_state = 42

            #Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=test_fraction, random_state=start_state
                )

        if k == 'poly':
            d = st.number_input("Degree of the polynomial kernel function", value=3)


        if k == 'rbf'or'sigmoid':
            classifier_svm = svm.SVC(C=c, kernel=k, gamma=g)

        if k == 'linear':
            classifier_svm = svm.SVC(C=c, kernel=k)

        classifier_svm.fit(X_train, y_train)

        y_pred = classifier_svm.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        accu = (conf_mat[0,0]+conf_mat[1,1])/sum(sum(conf_mat))

        st.write(f'The accuracy of the algorithm is {accu:.2%}')

        st.write('Here is the confusion matrix')
        ConfusionMatrixDisplay.from_estimator(classifier_svm, X_test, y_test)
        st.pyplot()

    with tab6:
        st.markdown('#### Personalized Heart Disease Prediction')
        st.markdown('**Input Your Health Data for a Customized Risk Assessment**')
        st.markdown('''In this section, we offer you a personalized heart disease prediction \
         experience. By filling out a simple form with your health-related information, \
         you can get an immediate assessment of your heart disease risk using our advanced \
         Random Forest algorithm.''')

        with st.form(key='patient_info_form'):
            st.write("Patient Information Form")

            age = st.number_input('Age', min_value=0, max_value=120, step=1)

            sex = st.selectbox('Sex', options=['Male', 'Female'])

            cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
            cp = st.selectbox('Chest Pain Type', options=cp_options)

            trestbps = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=0)

            chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0)

            fbs_options = ['True', 'False']
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=fbs_options)

            restecg_options = ['Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy']
            restecg = st.selectbox('Resting Electrocardiographic Results', options=restecg_options)

            thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)

            exang_options = ['No', 'Yes']
            exang = st.selectbox('Exercise Induced Angina', options=exang_options)

            oldpeak = st.number_input('ST depression induced by exercise relative to rest')

            slope_options = ['Upsloping', 'Flat', 'Downsloping']
            slope = st.selectbox('The Slope of The Peak Exercise ST Segment', options=slope_options)

            ca = st.number_input('Number of Major Vessels (0-3) Colored by Flourosopy', min_value=0, max_value=3, step=1)

            thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
            thal = st.selectbox('Thalassemia', options=thal_options)


            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            # Create a dictionary with the input data
            patient_data = {
                'Age': age,
                'Sex': 0 if sex == 'Male' else 1,
                'Chest Pain Type': cp_options.index(cp),
                'Resting Blood Pressure': trestbps,
                'Serum Cholestoral': chol,
                'Fasting Blood Sugar': 1 if fbs == 'True' else 0,
                'Resting ECG': restecg_options.index(restecg),
                'Max Heart Rate': thalach,
                'Exercise Induced Angina': 0 if exang == 'No' else 1,
                'Oldpeak': oldpeak,
                'Slope': slope_options.index(slope),
                'Major Vessels': ca,
                'Thalassemia': thal_options.index(thal)
            }

            patient_data_values = list(patient_data.values())
            patient_data_array = np.array(patient_data_values)
            patient_data_array = patient_data_array.reshape(1, -1)

            st.write("Patient data submitted successfully!")

            test_fraction = 0.2
            start_state = 42
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=test_fraction, random_state=start_state
                )
            classifier_rf = RandomForestClassifier(max_depth=5,
                                               n_estimators=100, oob_score=True)
            classifier_rf.fit(X_train, y_train)

            y_pred = classifier_rf.predict(patient_data_array)

            if y_pred == 0:
                st.markdown('''
                #### The predicted diagnosis is: :green[No heart disease]
                ''')

            else:
                st.markdown('''
                #### The predicted diagnosis is: :red[Heart disease]
                ''')

        with st.expander("**Important Disclaimer**"):
            st.markdown('''
            Please be aware that the predictions made by this model are for \
            informational purposes only and should not be taken as a substitute for \
            professional medical advice, diagnosis, or treatment.

            If you have concerns about your heart health or if the app's prediction \
            indicates a potential risk, we strongly encourage you to consult with a doctor \
            or healthcare provider. They can offer a thorough evaluation based on a complete \
            medical history, physical examination, and any necessary diagnostic tests.
            ''')


    with tab7:
        st.markdown('#### Our Journey to Heart Health Awareness')
        st.markdown('''As we reach the end of this interactive experience, \
        I'd like to take a moment to reflect on what we've accomplished and share a \
        bit about myself, the person behind this application.''')
        st.markdown(' ')
        st.markdown('**App Summary**')

        st.markdown('''
        This web app was designed with the goal of raising awareness about heart disease, the leading cause of mortality worldwide. Through interactive tools and machine learning models, we've explored various aspects of heart disease prediction. Key features of this app include:

        - **Educational Insights:** Providing valuable information about heart disease, its risk factors, and preventative measures.
        - **Interactive Analysis:** Allowing users to experiment with different machine learning models to understand how they predict heart disease based on clinical data.
        - **Personalized Prediction:** Offering a tool for users to input their health data and receive a personalized risk assessment for heart disease.

        I hope this application has been informative and empowering, shedding light on the significance of heart health and the potential of data-driven approaches in healthcare.
        ''')

        st.markdown('**About the Developer**')

        st.markdown('''
        Hello! I'm Yunlu Zhang, the developer of this web application. \
        I am a third-year Ph.D. student from department of Chemical Engineering and Materials Science(CHEMS).\
        With a passion for data science and healthcare technology, I develped this project \
        to help people understand the heart disease better and explore the application of machine learning tool\
        in predicting the dianosis result.

        Thank you for joining me on this journey! ''')
